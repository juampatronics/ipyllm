# ipyllm

`ipyllm` is an IPython extension for discussing your live Python session with
models supported by the [`llm` package](https://llm.datasette.io/). Ask
questions about your variables or REPL history, request plots, and have the
model run code in the active IPython shell.

Start with the [installation guide](installation.md), then see
[usage and commands](usage.md).

!!! warning "Code runs with your permissions"
    Model-requested code executes in the live Python process. It is not
    sandboxed and can access files, the network, and session objects. Enable
    approval when you want to review code before execution; approval is not a
    security boundary.
