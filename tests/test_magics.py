from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import llm
from IPython.core.error import UsageError
from IPython.core.interactiveshell import InteractiveShell

from ipyllm import load_ipython_extension, unload_ipython_extension
from ipyllm.magics import IpyllmMagics


class FakeResponse:
    def text(self):
        return "model response"


class FakeConversation:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.prompt = None
        self.system = None

    def chain(self, prompt, *, system):
        self.prompt = prompt
        self.system = system
        return FakeResponse()


class FakeModel:
    def conversation(self, **kwargs):
        self.created_conversation = FakeConversation(**kwargs)
        return self.created_conversation


class IpyllmMagicsTests(unittest.TestCase):
    def setUp(self):
        self.shell = InteractiveShell.instance()
        self.shell.user_ns["visible_test_value"] = [1, 2, 3]
        self.shell.user_ns["_hidden_test_value"] = "secret"
        self.magics = IpyllmMagics(self.shell)
        self.fake_model = FakeModel()
        self.model_patcher = patch("ipyllm.magics.llm.get_model", return_value=self.fake_model)
        self.get_model = self.model_patcher.start()
        self.addCleanup(self.model_patcher.stop)

    def tearDown(self):
        self.shell.user_ns.pop("visible_test_value", None)
        self.shell.user_ns.pop("_hidden_test_value", None)

    def test_extension_registers_magic(self):
        load_ipython_extension(self.shell)
        self.assertIsNotNone(self.shell.find_line_magic("llm"))
        unload_ipython_extension(self.shell)
        self.assertIsNone(self.shell.find_line_magic("llm"))

    def test_prompt_includes_only_namespace_names_and_types(self):
        self.magics.llm("Tell me about visible_test_value")

        prompt = self.fake_model.created_conversation.prompt
        self.assertIn("visible_test_value: list", prompt)
        self.assertNotIn("[1, 2, 3]", prompt)
        self.assertNotIn("_hidden_test_value", prompt)
        self.assertIn("Tell me about visible_test_value", prompt)
        tools = self.fake_model.created_conversation.options["tools"]
        self.assertEqual([tool.__name__ for tool in tools][-1], "execute_code")
        for tool in tools:
            llm.Tool.function(tool)

    def test_model_request_executes_in_live_namespace(self):
        response = self.magics.execute_code("visible_test_value.append(4)")

        self.assertEqual(self.shell.user_ns["visible_test_value"], [1, 2, 3, 4])
        self.assertEqual(response, "Execution completed successfully.")

    def test_execution_errors_are_returned_to_the_model(self):
        response = self.magics.execute_code("raise ValueError('broken')")

        self.assertIn("Execution failed: ValueError: broken", response)

    def test_variable_inspection_is_bounded(self):
        class LargeRepresentation:
            def __repr__(self):
                return "x" * 10_000

        self.shell.user_ns["large_test_value"] = LargeRepresentation()
        self.addCleanup(self.shell.user_ns.pop, "large_test_value", None)

        response = self.magics.inspect_variable("large_test_value")
        self.assertEqual(
            len(response),
            4_000
            + len("large_test_value: LargeRepresentation\n")
            + len("... (truncated)"),
        )

    def test_history_range_is_read_and_bounded(self):
        history = self.shell.history_manager
        entries = [(1, number, f"value = {number}") for number in range(40)]
        with patch.object(history, "get_range_by_str", return_value=iter(entries)) as get_range:
            response = self.magics.get_history("1-40")

        get_range.assert_called_once_with("1-40", raw=True)
        self.assertIn("In [1:0]:", response)
        self.assertNotIn("In [1:30]:", response)

    def test_approval_mode_requires_exact_yes(self):
        self.magics.require_approval = True
        call = SimpleNamespace(name="execute_code", arguments={"code": "x = 1"})

        with patch("builtins.input", return_value="no"):
            with self.assertRaises(llm.CancelToolCall):
                self.magics._before_tool_call(None, call)
        with patch("builtins.input", return_value="yes"):
            self.magics._before_tool_call(None, call)

    def test_approval_can_be_overridden_and_commands_validate(self):
        self.magics.require_approval = True
        self.magics._handle_command("--approval off")
        self.assertFalse(self.magics._current_approval_policy())

        with self.assertRaises(UsageError):
            self.magics._handle_command("--approval sometimes")

    def test_help_command_does_not_send_a_prompt(self):
        self.assertTrue(self.magics._handle_command("--help"))
        self.assertIsNone(self.magics._conversation)

    def test_reset_discards_conversation(self):
        self.magics._get_conversation()
        self.assertIsNotNone(self.magics._conversation)

        self.magics.reset()

        self.assertIsNone(self.magics._conversation)


if __name__ == "__main__":
    unittest.main()
