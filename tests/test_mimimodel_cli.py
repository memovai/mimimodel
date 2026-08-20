from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mimimodel_cli as cli


def tool(name="gpio_on", properties=None, required=None):
    properties = properties or {
        "pin": {"type": "integer", "description": "GPIO pin number"}
    }
    return {
        "name": name,
        "description": "Controls a device",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required if required is not None else list(properties),
        },
    }


class ToolValidationTest(unittest.TestCase):
    def test_accepts_supported_schema(self):
        checked = cli.validate_tools([tool()])
        self.assertEqual(checked[0]["name"], "gpio_on")

    def test_accepts_function_wrapper(self):
        checked = cli.validate_tools([{"type": "function", "function": tool()}])
        self.assertEqual(checked[0]["name"], "gpio_on")

    def test_rejects_duplicate_names(self):
        with self.assertRaisesRegex(cli.CliError, "duplicate tool name"):
            cli.validate_tools([tool(), tool()])

    def test_rejects_types_the_decoder_cannot_emit(self):
        bad = tool(properties={"enabled": {"type": "boolean"}})
        with self.assertRaisesRegex(cli.CliError, "unsupported type"):
            cli.validate_tools([bad])

    def test_rejects_unknown_required_parameter(self):
        bad = tool(required=["missing"])
        with self.assertRaisesRegex(cli.CliError, "unknown parameters"):
            cli.validate_tools([bad])

    def test_enforces_firmware_tool_limit(self):
        tools = [tool(f"tool_{index}", {}, []) for index in range(cli.MAX_TOOLS + 1)]
        with self.assertRaisesRegex(cli.CliError, "at most 16 tools"):
            cli.validate_tools(tools)

    def test_repository_demo_profile_is_valid(self):
        path = Path(__file__).parents[1] / "examples" / "tools" / "demo.json"
        tools = cli.load_tools_file(path)
        self.assertEqual(len(tools), 3)
        payload = cli.build_serial_payload("Turn on the flashlight.", tools)
        self.assertLessEqual(len(payload.encode("utf-8")), cli.MAX_SERIAL_LINE)


class SerialProtocolTest(unittest.TestCase):
    def test_prefix_hash_includes_system_prompt(self):
        tools = [tool()]
        self.assertNotEqual(cli.prefix_hash(tools), cli.prefix_hash(tools, "Be concise"))

    def test_builds_per_request_tool_payload(self):
        payload = cli.build_serial_payload("line one\nline two", [tool()], "system")
        system, query, tools_json = payload.split("\t")
        self.assertEqual(system, "system")
        self.assertEqual(query, "line one\x1eline two")
        self.assertEqual(json.loads(tools_json)[0]["name"], "gpio_on")

    def test_rejects_payload_larger_than_firmware_buffer(self):
        with self.assertRaisesRegex(cli.CliError, "firmware limit"):
            cli.build_serial_payload("x" * (cli.MAX_SERIAL_LINE + 1))

    def test_parses_device_call_and_timing(self):
        output = (
            '[needle] call: [{"name":"gpio_on","arguments":{"pin":5}}] (1234 ms)\r\n'
            '[needle] prefix cache: hit\r\n'
            '[needle] total 1234 ms | prefill 12 tok 1.50 tok/s | decode 8 tok 1.25 tok/s\r\n> '
        )
        parsed = cli.parse_device_output(output)
        self.assertEqual(parsed["calls"][0]["arguments"]["pin"], 5)
        self.assertTrue(parsed["prefix_reused"])
        self.assertEqual(parsed["timing"]["total_ms"], 1234)
        self.assertEqual(parsed["timing"]["decode_tps"], 1.25)


class ProfileCommandTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name) / "config"
        self.runtime = Path(self.temp.name) / "runtime"
        self.env = patch.dict(
            os.environ,
            {"MIMIMODEL_HOME": str(self.home), "MIMIMODEL_RUNTIME": str(self.runtime)},
        )
        self.env.start()
        self.tools_file = Path(self.temp.name) / "tools.json"
        self.tools_file.write_text(json.dumps([tool()]), encoding="utf-8")

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_first_import_activates_named_profile(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli.main([
                "tools", "import", str(self.tools_file), "--profile", "home"
            ])
        self.assertEqual(result, 0)
        config = cli.load_config()
        self.assertEqual(config["active_profile"], "home")
        self.assertEqual(config["profiles"]["home"][0]["name"], "gpio_on")

    def test_add_requires_replace_for_existing_name(self):
        cli.main(["tools", "import", str(self.tools_file)])
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            result = cli.main(["tools", "add", str(self.tools_file)])
        self.assertEqual(result, 2)
        self.assertIn("--replace", stderr.getvalue())

    def test_remove_last_tool_removes_profile(self):
        cli.main(["tools", "import", str(self.tools_file), "--profile", "home"])
        result = cli.main(["tools", "remove", "gpio_on", "--profile", "home"])
        self.assertEqual(result, 0)
        config = cli.load_config()
        self.assertNotIn("home", config["profiles"])
        self.assertEqual(config["active_profile"], "default")


if __name__ == "__main__":
    unittest.main()
