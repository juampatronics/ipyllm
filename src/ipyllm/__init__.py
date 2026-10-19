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


def main() -> None:
    print("Load this package in IPython with `%load_ext ipyllm`.")
