# Copyright 2026 Juan Pablo de la Cruz Gutiérrez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import patch

from IPython.core.interactiveshell import InteractiveShell
from IPython.terminal.embed import InteractiveShellEmbed  # noqa: T100
from traitlets.config import Config

from ipyllm import embed


class EmbedTests(unittest.TestCase):
    def _enter_embedded_shell(self, config: Config) -> InteractiveShellEmbed:
        shells = []
        original_call = InteractiveShellEmbed.__call__

        def enter(self, *args, **kwargs):
            shells.append(self)
            self.dummy_mode = True
            return original_call(self, *args, **kwargs)

        with patch.object(InteractiveShellEmbed, "__call__", enter):
            embed(config=config)

        self.assertEqual(len(shells), 1)
        return shells[0]

    def test_embedded_shell_has_llm_magics(self):
        shell = self._enter_embedded_shell(Config())

        self.assertIsNotNone(shell.find_line_magic("llm"))
        self.assertIsNotNone(shell.find_cell_magic("llm"))

    def test_embed_restores_previous_shell_instance(self):
        original = InteractiveShell.instance()

        self._enter_embedded_shell(Config())

        self.assertIs(InteractiveShell.instance(), original)

    def test_embed_applies_user_config(self):
        config = Config()
        config.InteractiveShellEmbed.term_title = True

        shell = self._enter_embedded_shell(config)

        self.assertTrue(shell.term_title)


if __name__ == "__main__":
    unittest.main()
