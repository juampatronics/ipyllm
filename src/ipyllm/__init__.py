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

"""Interact with LLM models from an IPython session."""

from .magics import IpyllmMagics


def load_ipython_extension(ipython) -> None:
    """Register the IPython magics provided by this package."""
    ipython.register_magics(IpyllmMagics)


def unload_ipython_extension(ipython) -> None:
    """Unregister the magics and clear their in-memory state."""
    manager = ipython.magics_manager
    magics = manager.registry.pop(IpyllmMagics.__name__, None)
    if magics is not None:
        magics.reset()
        for kind, registered in magics.magics.items():
            for name, function in registered.items():
                if manager.magics[kind].get(name) is function:
                    del manager.magics[kind][name]


def embed(header: str = "", compile_flags: int | None = None, **kwargs) -> None:
    """Embed an IPython shell at the current point with ipyllm preloaded.

    Behaves like ``IPython.terminal.embed``: the shell sees the caller's
    local variables, and the script resumes when the user exits. The
    difference is that ipyllm is loaded before the first prompt, so ``%llm``
    is available right away. Extra keyword arguments are passed to the
    embedded shell, for example ``config``.
    """
    import sys

    from IPython.core.interactiveshell import InteractiveShell
    from IPython.terminal.embed import InteractiveShellEmbed
    from IPython.terminal.ipapp import load_default_config

    config = kwargs.get("config")
    if config is None:
        config = load_default_config()
        config.InteractiveShellEmbed = config.TerminalInteractiveShell
        kwargs["config"] = config

    using = kwargs.get("using", "sync")
    colors = kwargs.pop("colors", "nocolor")
    if using:
        kwargs["config"].update(
            {
                "TerminalInteractiveShell": {
                    "loop_runner": using,
                    "colors": colors,
                    "autoawait": using != "sync",
                }
            }
        )

    saved_shell_instance = InteractiveShell._instance
    if saved_shell_instance is not None:
        type(saved_shell_instance).clear_instance()

    ps1 = None
    ps2 = None
    try:
        ps1 = sys.ps1
        ps2 = sys.ps2
    except AttributeError:
        pass

    frame = sys._getframe(1)
    location_id = f"{frame.f_code.co_filename}:{frame.f_lineno}"
    try:
        shell = InteractiveShellEmbed.instance(_init_location_id=location_id, **kwargs)
        shell.extension_manager.load_extension("ipyllm")
        shell(
            header=header,
            stack_depth=2,
            compile_flags=compile_flags,
            _call_location_id=location_id,
        )
    finally:
        InteractiveShellEmbed.clear_instance()
        if saved_shell_instance is not None:
            for subclass in type(saved_shell_instance)._walk_mro():
                subclass._instance = saved_shell_instance
        if ps1 is not None:
            sys.ps1 = ps1
            sys.ps2 = ps2

def main() -> None:
    print("Load this package in IPython with `%load_ext ipyllm`.")
