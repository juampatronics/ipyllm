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

Typing ``@@`` instead triggers a fuzzy search over the whole current
directory tree (via ``rapidfuzz``); the chosen match replaces the ``@@``
trigger and the loose query with a single ``@`` reference.
"""

import os
import re

from rapidfuzz import fuzz, process, utils


def _word_at(line: str, index: int) -> bool:
    """Return True when the character at ``index`` starts a word in ``line``."""
    return index == 0 or line[index - 1].isspace()


def _completion(text: str, kind: str = "file"):
    from IPython.core.completer import SimpleCompletion

    return SimpleCompletion(text=text, type=kind)


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


# Directories that never hold files a user would reference; skipped while
# scanning for the fuzzy index.
_IGNORED_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".idea",
    "site-packages",
    "dist-packages",
}
_MAX_INDEXED_ENTRIES = 20_000
_SCORE_CUTOFF = 45
_DEFAULT_LIMIT = 25

# The double-@ trigger followed by a loose query (spaces allowed) up to the
# cursor, optionally inside quotes; anchored so the query ends at the cursor.
_FUZZY_TRIGGER = re.compile(r"@@([^'\"\n]*)$")

_index_root: str | None = None
_index_paths: list[str] = []


def _scan_tree(root: str) -> list[str]:
    """Collect relative paths under ``root``: files, and dirs with a trailing ``/``."""
    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORED_DIRS)
        base = os.path.relpath(dirpath, root)
        if base == os.curdir:
            base = ""
        for name in sorted(filenames):
            paths.append(os.path.join(base, name).replace(os.sep, "/") if base else name)
        if len(paths) >= _MAX_INDEXED_ENTRIES:
            return paths
        for name in dirnames:
            rel = os.path.join(base, name) if base else name
            paths.append(rel.replace(os.sep, "/") + "/")
    return paths


def _fuzzy_search(query: str, limit: int) -> list[tuple[str, str]]:
    """Fuzzy-match ``query`` against the current directory tree.

    Returns ``(path, kind)`` pairs best match first; kind is ``"dir"`` when
    the path is a directory (marked with a trailing ``/``).
    """
    global _index_root, _index_paths
    root = os.getcwd()
    if root != _index_root:  # first use or the working directory changed
        _index_root = root
        _index_paths = _scan_tree(root)

    if not query.strip():
        return [(p, "dir" if p.endswith("/") else "file") for p in _index_paths[:limit]]

    hits = process.extract(
        query,
        _index_paths,
        scorer=fuzz.WRatio,
        processor=utils.default_process,
        limit=limit,
        score_cutoff=_SCORE_CUTOFF,
    )
    return [
        (_index_paths[i], "dir" if _index_paths[i].endswith("/") else "file")
        for _, _score, i in hits
    ]


class FuzzyFileMatcher:
    """An IPython completion matcher (API v2) for fuzzy ``@@`` references.

    A no-op (empty result, other matchers proceed) unless the text before the
    cursor ends with a word-leading ``@@``. In that case the query after the
    trigger (which may contain spaces) is fuzzy-matched against the whole
    current directory tree, and each candidate replaces the trigger and query
    with a single-``@`` reference so ``%llm`` resolves it as a file.
    """

    matcher_api_version = 2
    matcher_identifier = "ipyllm.fuzzy_file_matcher"

    def __init__(self) -> None:
        # _get_matcher_id falls back to ``__qualname__`` when reading the id,
        # which is evaluated eagerly; provide it so instances resolve cleanly.
        self.__qualname__ = "ipyllm.completer.FuzzyFileMatcher"

    def __call__(self, context) -> dict:
        text = context.text_until_cursor
        match = _FUZZY_TRIGGER.search(text)
        if not match:
            return self._result([])
        if text[: match.start()] and not text[match.start() - 1].isspace():
            return self._result([])  # not word-leading; e.g. a double-@ in a word

        query = match.group(1)
        limit = context.limit or _DEFAULT_LIMIT
        hits = _fuzzy_search(query, limit)
        completions = [_completion("@" + path, kind) for path, kind in hits]
        return self._result(
            completions,
            matched_fragment=text[match.start():],  # replace "@@query"
        )

    @staticmethod
    def _result(completions: list, matched_fragment: str | None = None) -> dict:
        result = {
            "completions": completions,
            "suppress": bool(completions),
            "ordered": True,
        }
        if matched_fragment is not None:
            result["matched_fragment"] = matched_fragment
        return result


fuzzy_file_matcher = FuzzyFileMatcher()


def install_completer(shell) -> None:
    """Register the ``@`` and ``@@`` file matchers on ``shell``'s completer."""
    completer = getattr(shell, "Completer", None)
    if completer is None:
        return
    for matcher in (at_file_matcher, fuzzy_file_matcher):
        if matcher not in completer.custom_matchers:
            completer.custom_matchers.append(matcher)


def uninstall_completer(shell) -> None:
    """Remove the ``@`` and ``@@`` file matchers from ``shell``, if present."""
    completer = getattr(shell, "Completer", None)
    if completer is None:
        return
    for matcher in (at_file_matcher, fuzzy_file_matcher):
        try:
            completer.custom_matchers.remove(matcher)
        except ValueError:
            pass
