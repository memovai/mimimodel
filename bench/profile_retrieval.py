"""Profile the ESP32 engine's BM25 tool pruning on mobile-actions."""
import argparse
import heapq
import json
import os
import struct
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from needle_np import RefTokenizer  # noqa: E402
from mobile_actions import (build_turn, common_bm25_tools, ensure_dataset, expected,
                            to_needle_tools)  # noqa: E402

DT_RAW = 4
HEADER_BYTES = 30 * 4
RECORD_BYTES = 44
META_SPACE = "▁"


def load_tokenizer(path):
    """Read only the RAW tokenizer record; do not dequantize model matrices."""
    with open(path, "rb") as handle:
        blob = handle.read()
    num_tensors, codebook_count = struct.unpack_from("<II", blob, 4)
    directory = HEADER_BYTES + codebook_count * 4
    for index in range(num_tensors):
        record = directory + index * RECORD_BYTES
        dtype = blob[record]
        offset, nbytes = struct.unpack_from("<QQ", blob, record + 20)
        if dtype == DT_RAW:
            return RefTokenizer(blob[offset:offset + nbytes])
    raise ValueError("model has no RAW tokenizer record")


def fast_bpe_count(tokenizer, segment):
    """Equivalent to RefTokenizer._bpe, using a heap instead of full rescans."""
    if not segment:
        return 0
    symbols = list(segment)
    size = len(symbols)
    previous = [index - 1 for index in range(size)]
    following = [index + 1 for index in range(size)]
    following[-1] = -1
    alive = [True] * size
    heap = []

    def push(left):
        if left < 0 or not alive[left]:
            return
        right = following[left]
        if right < 0 or not alive[right]:
            return
        piece_id = tokenizer.p2id.get(symbols[left] + symbols[right])
        if piece_id is not None:
            heapq.heappush(heap, (-tokenizer.scores[piece_id], left, right,
                                  symbols[left], symbols[right]))

    for index in range(size - 1):
        push(index)
    while heap:
        _, left, right, left_symbol, right_symbol = heapq.heappop(heap)
        if (not alive[left] or not alive[right] or following[left] != right
                or symbols[left] != left_symbol or symbols[right] != right_symbol):
            continue
        symbols[left] += symbols[right]
        alive[right] = False
        following[left] = following[right]
        if following[right] >= 0:
            previous[following[right]] = left
        push(previous[left])
        push(left)

    count = 0
    index = 0
    while index >= 0:
        symbol = symbols[index]
        if symbol in tokenizer.p2id:
            count += 1
        elif tokenizer.byte_fallback:
            count += len(symbol.encode("utf-8"))
        else:
            count += 1
        index = following[index]
    return count


def fast_encode_count(tokenizer, text):
    if not text:
        return 0
    escaped = text.replace(" ", META_SPACE)
    if tokenizer.add_dummy:
        escaped = META_SPACE + escaped
    count = 0
    buffer = []
    index = 0
    while index < len(escaped):
        marker = next((item for item in tokenizer.markers
                       if escaped.startswith(item, index)), None)
        if marker is None:
            buffer.append(escaped[index])
            index += 1
            continue
        count += fast_bpe_count(tokenizer, "".join(buffer)) + 1
        buffer = []
        index += len(marker)
    return count + fast_bpe_count(tokenizer, "".join(buffer))


def native_bm25_selection(query, tools, tokenizer, budget):
    compact = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    if fast_encode_count(tokenizer, compact) <= budget or len(tools) <= 2:
        return tools
    ranked = common_bm25_tools(query, tools, len(tools))
    for keep in range(len(tools), 2, -1):
        candidate = json.dumps(ranked[:keep], separators=(",", ":"), ensure_ascii=False)
        if fast_encode_count(tokenizer, candidate) <= budget:
            return ranked[:keep]
    return ranked[:2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(ROOT, "model", "needle2.cact"))
    parser.add_argument("--budget", type=int, default=180)
    parser.add_argument("--results", help="optional engine rows for correlating retrieval misses")
    parser.add_argument("--out", help="optional JSON report")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    records = [row for row in ensure_dataset() if row["metadata"] == "eval"]
    result_rows = None
    if args.results:
        with open(args.results) as handle:
            result_rows = json.load(handle)
        if isinstance(result_rows, dict):
            result_rows = result_rows.get("rows", [])
        if len(result_rows) != len(records):
            raise ValueError("result row count does not match eval dataset")

    kept_counts = Counter()
    misses = []
    under_with_miss = []
    under_with_recall = []
    miss_outcomes = Counter()
    max_prefix = (0, -1)
    for index, record in enumerate(records):
        system, query = build_turn(record)
        selected = native_bm25_selection(query, to_needle_tools(record["tools"]),
                                         tokenizer, args.budget)
        selected_json = json.dumps(selected, separators=(",", ":"), ensure_ascii=False)
        prefix = (f"<|im_start|>system\n{system}<|im_end|>\n" if system else "")
        prefix += f"<|im_start|>user\n<tools>{selected_json}</tools>"
        prefix_tokens = 1 + fast_encode_count(tokenizer, prefix)
        if prefix_tokens > max_prefix[0]:
            max_prefix = (prefix_tokens, index)
        selected_names = [tool["name"] for tool in selected]
        expected_names = [call["name"] for call in expected(record)]
        kept_counts[len(selected)] += 1
        recall = set(expected_names) <= set(selected_names)
        if not recall:
            misses.append({
                "row": index,
                "expected": expected_names,
                "selected": selected_names,
            })
            if result_rows is not None:
                got_count = len(result_rows[index].get("got") or [])
                if got_count < len(expected_names):
                    miss_outcomes["under_call"] += 1
                elif got_count > len(expected_names):
                    miss_outcomes["over_call"] += 1
                elif result_rows[index].get("name_ok"):
                    miss_outcomes["names_correct"] += 1
                else:
                    miss_outcomes["wrong_name"] += 1
        if result_rows is not None:
            got_count = len(result_rows[index].get("got") or [])
            if got_count < len(expected_names):
                (under_with_recall if recall else under_with_miss).append(index)

    report = {
        "total": len(records),
        "budget_tokens": args.budget,
        "kept_tool_counts": dict(sorted(kept_counts.items())),
        "full_expected_sequence_recall": len(records) - len(misses),
        "retrieval_misses": len(misses),
        "retrieval_miss_rows": [item["row"] for item in misses],
        "retrieval_miss_examples": misses[:20],
        "max_prefix_tokens": max_prefix[0],
        "max_prefix_row": max_prefix[1],
    }
    if result_rows is not None:
        report["under_call_with_retrieval_miss"] = len(under_with_miss)
        report["under_call_despite_full_recall"] = len(under_with_recall)
        report["under_call_with_retrieval_miss_rows"] = under_with_miss[:20]
        report["under_call_despite_full_recall_rows"] = under_with_recall[:20]
        report["retrieval_miss_outcomes"] = dict(miss_outcomes)

    print(f"native BM25 budget={args.budget}: expected sequence retained "
          f"{report['full_expected_sequence_recall']}/{report['total']}")
    print("kept tools:", ", ".join(f"{key}={value}" for key, value in kept_counts.items()))
    print(f"max prefix: {max_prefix[0]} tokens at row {max_prefix[1]}")
    if result_rows is not None:
        print(f"under-call split: retrieval miss={len(under_with_miss)}, "
              f"full recall={len(under_with_recall)}")
        print("retrieval-miss outcomes:", ", ".join(
            f"{key}={value}" for key, value in miss_outcomes.items()))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print("report ->", args.out)


if __name__ == "__main__":
    main()
