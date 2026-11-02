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

"""Autocomplete local file and directory names for ``@`` prompt references.

When the user types ``@`` at the start of a word, IPython offers the entries
of the referenced directory (files and subdirectories, the latter with a
trailing ``/``) so paths can be assembled one tab at a time.
"""

import os


def _word_at(line: str, index: int) -> bool:
    """Return True when the character at ``index`` starts a word in ``line``."""
    return index == 0 or line[index - 1].isspace()


def _completion(text: str):
    from IPython.core.completer import SimpleCompletion

    return SimpleCompletion(text=text, type="file")


def _list_matches(base: str, fragment: str) -> list:
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return []
    matches = []
    for entry in entries:
        if not entry.startswith(fragment):
            continue
        full = os.path.join(base, entry) if base != os.curdir else entry
        if os.path.isdir(full):
            matches.append(_completion(entry + "/"))
        else:
            matches.append(_completion(entry))
    return matches


class AtFileMatcher:
    """An IPython completion matcher (API v2) that completes ``@`` paths.

    A no-op (empty result, other matchers proceed) unless the token being
    completed is directly preceded by an ``@`` that starts a word. This keeps
    emails and other in-word ``@`` characters untouched.
    """

    matcher_api_version = 2
    matcher_identifier = "ipyllm.at_file_matcher"

    def __init__(self) -> None:
        # _get_matcher_id falls back to ``__qualname__`` when reading the id,
        # which is evaluated eagerly; provide it so instances resolve cleanly.
        self.__qualname__ = "ipyllm.completer.AtFileMatcher"

    def __call__(self, context) -> dict:
        token = context.token
        prefix = context.text_until_cursor
        before = prefix[:-len(token)] if token else prefix

        if not before.endswith("@"):
            return self._result([])
        at_index = len(before) - 1
        if not _word_at(before, at_index):
            return self._result([])

        if token:
            base = os.path.dirname(token) or os.curdir
            fragment = os.path.basename(token)
        else:
            base = os.curdir
            fragment = ""

        matches = _list_matches(base, fragment)
        return self._result(matches)

    @staticmethod
    def _result(completions: list) -> dict:
        return {
            "completions": completions,
            "suppress": bool(completions),
            "ordered": True,
        }


at_file_matcher = AtFileMatcher()


def install_completer(shell) -> None:
    """Register the ``@`` file matcher on ``shell``'s completer."""
    completer = getattr(shell, "Completer", None)
    if completer is None:
        return
    if at_file_matcher not in completer.custom_matchers:
        completer.custom_matchers.append(at_file_matcher)


def uninstall_completer(shell) -> None:
    """Remove the ``@`` file matcher from ``shell``'s completer, if present."""
    completer = getattr(shell, "Completer", None)
    if completer is None:
        return
    try:
        completer.custom_matchers.remove(at_file_matcher)
    except ValueError:
        pass
