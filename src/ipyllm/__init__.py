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
