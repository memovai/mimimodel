#!/usr/bin/env python3
"""Host CLI for configuring tool schemas and querying MimiModel over serial."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time


VERSION = "0.1.0"
MAX_TOOLS = 16
MAX_PARAMS = 12
MAX_SERIAL_LINE = 4094
DEFAULT_TIMEOUT = 700
TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
PARAM_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,46}$")
SUPPORTED_TYPES = {"string", "integer", "number"}


class CliError(Exception):
    pass


def config_dir():
    override = os.environ.get("MIMIMODEL_HOME")
    if override:
        return Path(override)
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "mimimodel"


def runtime_dir():
    override = os.environ.get("MIMIMODEL_RUNTIME")
    if override:
        return Path(override)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mimimodel"


def config_path():
    return config_dir() / "config.json"


def socket_path():
    return runtime_dir() / "daemon.sock"


def state_path():
    return runtime_dir() / "daemon.json"


def log_path():
    return runtime_dir() / "daemon.log"


def ensure_private_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def empty_config():
    return {"version": 1, "active_profile": "default", "profiles": {}}


def load_config():
    path = config_path()
    if not path.exists():
        return empty_config()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CliError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != 1:
        raise CliError(f"unsupported config format in {path}")
    if not isinstance(data.get("profiles"), dict):
        raise CliError(f"invalid profiles in {path}")
    return data


def save_config(data):
    ensure_private_dir(config_dir())
    path = config_path()
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.chmod(0o600)
    temp.replace(path)


def normalize_tool(tool):
    if (isinstance(tool, dict) and "function" in tool
            and set(tool).issubset({"type", "function"})
            and tool.get("type", "function") == "function"):
        tool = tool["function"]
    if not isinstance(tool, dict):
        raise CliError("each tool must be a JSON object")
    normalized = dict(tool)
    normalized.setdefault("description", "")
    normalized.setdefault("parameters", {"type": "object", "properties": {}})
    return normalized


def validate_tools(tools):
    if not isinstance(tools, list) or not tools:
        raise CliError("tool list must be a non-empty JSON array")
    if len(tools) > MAX_TOOLS:
        raise CliError(f"firmware supports at most {MAX_TOOLS} tools; got {len(tools)}")
    normalized = [normalize_tool(tool) for tool in tools]
    seen = set()
    for index, tool in enumerate(normalized):
        name = tool.get("name")
        if not isinstance(name, str) or not TOOL_NAME.fullmatch(name):
            raise CliError(f"tool {index}: name must match {TOOL_NAME.pattern}")
        if name in seen:
            raise CliError(f"duplicate tool name: {name}")
        seen.add(name)
        if not isinstance(tool.get("description"), str):
            raise CliError(f"tool {name}: description must be a string")
        parameters = tool.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise CliError(f"tool {name}: parameters.type must be object")
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        if not isinstance(properties, dict):
            raise CliError(f"tool {name}: parameters.properties must be an object")
        if len(properties) > MAX_PARAMS:
            raise CliError(
                f"tool {name}: firmware supports at most {MAX_PARAMS} parameters; got {len(properties)}"
            )
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            raise CliError(f"tool {name}: required must be an array of parameter names")
        unknown_required = set(required) - set(properties)
        if unknown_required:
            raise CliError(f"tool {name}: required references unknown parameters: "
                           f"{', '.join(sorted(unknown_required))}")
        for param_name, spec in properties.items():
            if not isinstance(param_name, str) or not PARAM_NAME.fullmatch(param_name):
                raise CliError(f"tool {name}: parameter name must match {PARAM_NAME.pattern}")
            if not isinstance(spec, dict) or spec.get("type") not in SUPPORTED_TYPES:
                got = spec.get("type") if isinstance(spec, dict) else None
                raise CliError(f"tool {name}.{param_name}: unsupported type {got!r}; "
                               f"use string, integer, or number")
    return normalized


def load_tools_file(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CliError(f"cannot read tools from {path}: {exc}") from exc
    if isinstance(data, dict) and "tools" in data:
        data = data["tools"]
    elif isinstance(data, dict):
        data = [data]
    return validate_tools(data)


def compact_tools(tools):
    return json.dumps(tools, separators=(",", ":"), ensure_ascii=False)


def tools_hash(tools):
    return hashlib.sha256(compact_tools(tools).encode("utf-8")).hexdigest()


def prefix_hash(tools, system=""):
    schema = compact_tools(tools) if tools is not None else "firmware-default"
    return hashlib.sha256(f"{schema}\0{system}".encode("utf-8")).hexdigest()


def active_profile(config, requested=None):
    return requested or config.get("active_profile") or "default"


def get_profile(config, requested=None, required=True):
    name = active_profile(config, requested)
    tools = config["profiles"].get(name)
    if tools is None and required:
        raise CliError(f"profile {name!r} does not exist; import tools first")
    return name, tools


def encode_field(text):
    return text.replace("\x1e", " ").replace("\t", " ").replace("\r\n", "\n").replace("\n", "\x1e")


def build_serial_payload(query, tools=None, system=""):
    if not isinstance(query, str) or not query.strip():
        raise CliError("query must not be empty")
    encoded_query = encode_field(query)
    encoded_system = encode_field(system)
    if tools is None:
        line = f"{encoded_system}\t{encoded_query}" if encoded_system else encoded_query
    else:
        checked = validate_tools(tools)
        line = f"{encoded_system}\t{encoded_query}\t{compact_tools(checked)}"
    size = len(line.encode("utf-8"))
    if size > MAX_SERIAL_LINE:
        raise CliError(f"serial request is {size} bytes; current firmware limit is "
                       f"{MAX_SERIAL_LINE}. Shorten descriptions or use fewer tools")
    return line


def parse_device_output(output):
    call = re.search(r"^\[needle\] call: (.+?) \(\d+ ms\)\r?$", output, re.MULTILINE)
    if not call:
        call = re.search(r"^\[needle\] call: (.+?)\r?$", output, re.MULTILINE)
    if not call:
        raise CliError("device response did not contain a tool call")
    try:
        calls = json.loads(call.group(1))
    except json.JSONDecodeError as exc:
        raise CliError(f"device returned invalid JSON: {call.group(1)}") from exc
    timing = re.search(
        r"\[needle\] total (\d+) ms \| prefill (\d+) tok ([0-9.]+) tok/s "
        r"\| decode (\d+) tok ([0-9.]+) tok/s", output
    )
    result = {"calls": calls}
    cache = re.search(r"\[needle\] prefix cache: (hit|miss)", output)
    if cache:
        result["prefix_reused"] = cache.group(1) == "hit"
    if timing:
        result["timing"] = {
            "total_ms": int(timing.group(1)),
            "prefill_tokens": int(timing.group(2)),
            "prefill_tps": float(timing.group(3)),
            "decode_tokens": int(timing.group(4)),
            "decode_tps": float(timing.group(5)),
        }
    return result


def find_serial_port():
    patterns = (
        "/dev/cu.usbmodem*",
        "/dev/cu.usbserial*",
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
    )
    ports = []
    for pattern in patterns:
        ports.extend(sorted(Path("/").glob(pattern.lstrip("/"))))
    unique = list(dict.fromkeys(str(port) for port in ports))
    if not unique:
        raise CliError("no ESP32 serial port found; pass --port explicitly")
    if len(unique) > 1:
        raise CliError("multiple serial ports found; pass --port: " + ", ".join(unique))
    return unique[0]


def read_json_line(sock):
    chunks = []
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    if not chunks:
        raise CliError("daemon closed the connection without a response")
    try:
        return json.loads(b"".join(chunks).split(b"\n", 1)[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError("invalid response from daemon") from exc


def daemon_request(request, timeout=5):
    sock_path = socket_path()
    if not sock_path.exists():
        raise CliError("daemon is not running")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(sock_path))
            sock.sendall(json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n")
            response = read_json_line(sock)
    except (OSError, socket.timeout) as exc:
        raise CliError(f"cannot reach daemon: {exc}") from exc
    if not response.get("ok"):
        raise CliError(response.get("error", "daemon request failed"))
    return response


def daemon_status():
    try:
        return daemon_request({"op": "status"}, timeout=1)
    except CliError:
        return None


def stop_daemon(quiet=False):
    status = daemon_status()
    if not status:
        if not quiet:
            print("daemon is not running")
        stale = socket_path()
        if stale.exists():
            stale.unlink()
        return
    daemon_request({"op": "shutdown"}, timeout=2)
    deadline = time.time() + 5
    while socket_path().exists() and time.time() < deadline:
        time.sleep(0.05)
    if not quiet:
        print("daemon stopped")


def ensure_daemon(requested_port=None):
    status = daemon_status()
    if status and (not requested_port or status.get("port") == requested_port):
        return status
    if status:
        stop_daemon(quiet=True)
    port = requested_port or find_serial_port()
    ensure_private_dir(runtime_dir())
    stale = socket_path()
    if stale.exists():
        stale.unlink()
    log_handle = log_path().open("a", encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve()), "_daemon", "--port", port]
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    log_handle.close()
    deadline = time.time() + 150
    while time.time() < deadline:
        if process.poll() is not None:
            tail = ""
            try:
                tail = log_path().read_text(encoding="utf-8")[-2000:]
            except OSError:
                pass
            raise CliError(f"daemon exited during startup\n{tail}")
        status = daemon_status()
        if status:
            return status
        time.sleep(0.2)
    process.terminate()
    raise CliError(f"timed out waiting for the board on {port}; see {log_path()}")


class BoardConnection:
    def __init__(self, port):
        try:
            import serial
        except ImportError as exc:
            raise CliError("pyserial is required; run: python3 -m pip install pyserial") from exc
        self.port = port
        self.started = time.time()
        self.last_prefix_hash = None
        self.serial = serial.Serial()
        self.serial.port = port
        self.serial.baudrate = 115200
        self.serial.timeout = 0.1
        self.serial.dtr = False
        self.serial.rts = False
        self.serial.open()
        self.serial.rts = True
        time.sleep(0.15)
        self.serial.rts = False
        boot = self.read_until("REPL ready", 120)
        if boot is None:
            self.serial.close()
            raise CliError(f"board on {port} did not reach the REPL")
        build = re.search(r"\[needle\] build:[^\r\n]*", boot)
        self.build = build.group(0) if build else None
        self.serial.reset_input_buffer()

    def close(self):
        if self.serial.is_open:
            self.serial.close()

    def read_until(self, marker, timeout):
        output = ""
        deadline = time.time() + timeout
        while time.time() < deadline:
            chunk = self.serial.read(4096)
            if chunk:
                output += chunk.decode("utf-8", "replace")
                if marker in output:
                    return output
        return None

    def run(self, line, request_prefix_hash, timeout):
        self.serial.reset_input_buffer()
        payload = (line + "\n").encode("utf-8")
        for offset in range(0, len(payload), 32):
            self.serial.write(payload[offset:offset + 32])
            self.serial.flush()
            time.sleep(0.03)
        output = self.read_until("\n> ", timeout)
        if output is None:
            raise CliError(f"device timed out after {timeout} seconds")
        same_requested_prefix = request_prefix_hash == self.last_prefix_hash
        self.last_prefix_hash = request_prefix_hash
        parsed = parse_device_output(output)
        parsed.setdefault("prefix_reused", same_requested_prefix)
        parsed["prefix_hash"] = request_prefix_hash
        return parsed


class DaemonHandler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            request = json.loads(self.rfile.readline().decode("utf-8"))
            op = request.get("op")
            board = self.server.board
            if op == "status":
                response = {
                    "ok": True,
                    "port": board.port,
                    "pid": os.getpid(),
                    "uptime_seconds": round(time.time() - board.started, 1),
                    "build": board.build,
                    "prefix_hash": board.last_prefix_hash,
                }
            elif op == "run":
                result = board.run(
                    request["line"],
                    request["prefix_hash"],
                    int(request.get("timeout", DEFAULT_TIMEOUT)),
                )
                response = {"ok": True, **result}
            elif op == "shutdown":
                response = {"ok": True}
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                raise CliError(f"unknown daemon operation: {op}")
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        self.wfile.write(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")


class DaemonServer(socketserver.UnixStreamServer):
    pass


def run_daemon(port):
    ensure_private_dir(runtime_dir())
    path = socket_path()
    if path.exists():
        path.unlink()
    board = BoardConnection(port)
    server = DaemonServer(str(path), DaemonHandler)
    server.board = board
    path.chmod(0o600)
    state_path().write_text(json.dumps({"pid": os.getpid(), "port": port}) + "\n", encoding="utf-8")

    def request_stop(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        board.close()
        if path.exists():
            path.unlink()
        if state_path().exists():
            state_path().unlink()


def tools_command(args):
    config = load_config()
    profile_name = active_profile(config, getattr(args, "profile", None))
    if args.tools_command == "import":
        tools = load_tools_file(args.path)
        first_profile = not config["profiles"]
        config["profiles"][profile_name] = tools
        if args.activate or first_profile:
            config["active_profile"] = profile_name
        save_config(config)
        print(f"imported {len(tools)} tools into profile {profile_name!r} ({tools_hash(tools)[:12]})")
    elif args.tools_command == "add":
        incoming = load_tools_file(args.path)
        existing = list(config["profiles"].get(profile_name, []))
        by_name = {tool["name"]: tool for tool in existing}
        for tool in incoming:
            if tool["name"] in by_name and not args.replace:
                raise CliError(f"tool {tool['name']!r} already exists; pass --replace")
            by_name[tool["name"]] = tool
        merged = validate_tools(list(by_name.values()))
        config["profiles"][profile_name] = merged
        save_config(config)
        print(f"profile {profile_name!r} now contains {len(merged)} tools")
    elif args.tools_command == "list":
        name, tools = get_profile(config, args.profile)
        print(f"profile: {name} ({tools_hash(tools)[:12]})")
        for tool in tools:
            properties = tool["parameters"].get("properties", {})
            required = set(tool["parameters"].get("required", []))
            params = ", ".join(
                f"{param}:{spec['type']}{'*' if param in required else ''}"
                for param, spec in properties.items()
            ) or "-"
            print(f"{tool['name']:<32} {params}")
    elif args.tools_command == "remove":
        name, tools = get_profile(config, args.profile)
        filtered = [tool for tool in tools if tool["name"] != args.name]
        if len(filtered) == len(tools):
            raise CliError(f"tool {args.name!r} is not in profile {name!r}")
        if filtered:
            config["profiles"][name] = filtered
        else:
            del config["profiles"][name]
            if config.get("active_profile") == name:
                config["active_profile"] = next(iter(config["profiles"]), "default")
        save_config(config)
        print(f"removed {args.name!r} from profile {name!r}")
    elif args.tools_command == "validate":
        if args.path:
            tools = load_tools_file(args.path)
            source = args.path
        else:
            source, tools = get_profile(config, args.profile)
        encoded = compact_tools(tools).encode("utf-8")
        print(f"valid: {source} · {len(tools)} tools · {len(encoded)} schema bytes · "
              f"sha256 {tools_hash(tools)[:12]}")
    elif args.tools_command == "use":
        if args.name not in config["profiles"]:
            raise CliError(f"profile {args.name!r} does not exist")
        config["active_profile"] = args.name
        save_config(config)
        print(f"active profile: {args.name}")
    elif args.tools_command == "profiles":
        active = config.get("active_profile")
        for name, tools in config["profiles"].items():
            marker = "*" if name == active else " "
            print(f"{marker} {name:<24} {len(tools)} tools  {tools_hash(tools)[:12]}")


def run_command(args):
    config = load_config()
    selected_tools = None
    profile_name = "firmware-default"
    if args.tools:
        selected_tools = load_tools_file(args.tools)
        profile_name = str(args.tools)
    elif args.profile or config["profiles"]:
        profile_name, selected_tools = get_profile(config, args.profile)
    line = build_serial_payload(args.query, selected_tools, args.system or "")
    request_prefix_hash = prefix_hash(selected_tools, args.system or "")
    status = ensure_daemon(args.port)
    print(f"device {status['port']} · profile {profile_name} · "
          f"prefix {request_prefix_hash[:12]}", file=sys.stderr)
    response = daemon_request(
        {"op": "run", "line": line, "prefix_hash": request_prefix_hash,
         "timeout": args.timeout},
        timeout=args.timeout + 5,
    )
    print(json.dumps(response["calls"], separators=(",", ":"), ensure_ascii=False))
    timing = response.get("timing")
    if timing:
        cache = "cache hit" if response.get("prefix_reused") else "cache miss"
        print(f"{cache} · {timing['total_ms'] / 1000:.1f}s · "
              f"prefill {timing['prefill_tokens']} tok @ {timing['prefill_tps']:.2f} tok/s · "
              f"decode {timing['decode_tps']:.2f} tok/s", file=sys.stderr)


def status_command():
    status = daemon_status()
    if not status:
        print("daemon: stopped")
        return
    print("daemon: running")
    print(f"port: {status['port']}")
    print(f"pid: {status['pid']}")
    print(f"uptime: {status['uptime_seconds']}s")
    print(f"firmware: {status.get('build') or 'unknown'}")
    print(f"prefix: {status.get('prefix_hash') or 'not used yet'}")


def build_parser():
    parser = argparse.ArgumentParser(prog="mimimodel")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    commands = parser.add_subparsers(dest="command", required=True)

    tools = commands.add_parser("tools", help="manage tool-schema profiles")
    tool_commands = tools.add_subparsers(dest="tools_command", required=True)

    import_cmd = tool_commands.add_parser("import", help="replace a profile from a JSON file")
    import_cmd.add_argument("path")
    import_cmd.add_argument("--profile", default=None)
    import_cmd.add_argument("--activate", action="store_true")

    add_cmd = tool_commands.add_parser("add", help="add tool schemas from a JSON file")
    add_cmd.add_argument("path")
    add_cmd.add_argument("--profile", default=None)
    add_cmd.add_argument("--replace", action="store_true")

    list_cmd = tool_commands.add_parser("list", help="list tools in a profile")
    list_cmd.add_argument("--profile", default=None)

    remove_cmd = tool_commands.add_parser("remove", help="remove one tool by name")
    remove_cmd.add_argument("name")
    remove_cmd.add_argument("--profile", default=None)

    validate_cmd = tool_commands.add_parser("validate", help="validate a file or profile")
    validate_cmd.add_argument("path", nargs="?")
    validate_cmd.add_argument("--profile", default=None)

    use_cmd = tool_commands.add_parser("use", help="select the active profile")
    use_cmd.add_argument("name")
    tool_commands.add_parser("profiles", help="list profiles")

    run = commands.add_parser("run", help="run one tool-calling query on the ESP32-S3")
    run.add_argument("query")
    run.add_argument("--profile")
    run.add_argument("--tools", help="temporary JSON tool file")
    run.add_argument("--system", default="")
    run.add_argument("--port")
    run.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)

    commands.add_parser("status", help="show daemon and board status")
    daemon = commands.add_parser("daemon", help="manage the serial daemon")
    daemon.add_argument("action", choices=("stop",))

    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_daemon":
        internal = argparse.ArgumentParser(prog="mimimodel _daemon")
        internal.add_argument("_daemon")
        internal.add_argument("--port", required=True)
        args = internal.parse_args(argv)
        try:
            run_daemon(args.port)
            return 0
        except CliError as exc:
            print(f"mimimodel: error: {exc}", file=sys.stderr)
            return 2
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "tools":
            tools_command(args)
        elif args.command == "run":
            run_command(args)
        elif args.command == "status":
            status_command()
        elif args.command == "daemon":
            stop_daemon()
        return 0
    except CliError as exc:
        print(f"mimimodel: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
