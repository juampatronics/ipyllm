# Installation

Install `ipyllm` into the same Python environment used by IPython:

```console
python -m pip install .
```

Run that command from the project checkout. The project requires Python 3.14 or
newer.

## Configure a model

`ipyllm` uses the model and provider configuration managed by
[`llm`](https://llm.datasette.io/). Configure a provider key, then choose a
default model:

```console
llm keys set PROVIDER
llm models
llm models default MODEL
```

For local Ollama models, the `llm-ollama` plugin is included. Install and start
Ollama, then select an installed model with `llm models default MODEL`.

## Install remote provider support

The optional `providers` dependency group installs plugins for Anthropic,
DeepSeek, Gemini, Grok, Mistral, OpenRouter, and Perplexity. In a checkout of
this project, install the group and run IPython in the resulting environment:

```console
uv sync --group providers
uv run ipython
```

`uv run` uses the project environment, so you do not need to activate it
manually. If you prefer an activated shell, use `source .venv/bin/activate`
on macOS/Linux or `.venv\Scripts\activate` on Windows, then run `ipython`.

The group installs these plugins: `llm-anthropic`, `llm-deepseek`,
`llm-gemini`, `llm-grok`, `llm-mistral`, `llm-openrouter`, and
`llm-perplexity`. If you installed `ipyllm` into another environment rather
than syncing this checkout, install the plugin package(s) you need into that
same environment, for example:

```console
python -m pip install llm-anthropic llm-gemini
```

Configure credentials for each provider using the key name documented by its
plugin. You can store a key with `llm keys set PROVIDER`, or set the provider's
documented environment variable in the shell before starting IPython. For
example, `llm keys set anthropic` stores an Anthropic key; environment
variables must be available to the IPython process.

Then list available models and choose one:

```console
llm models
llm models default MODEL
```

See the [`llm` remote API plugin directory](https://llm.datasette.io/en/stable/plugins/directory.html#remote-apis)
for the supported providers and links to plugin-specific setup instructions.
The [`llm` API key guide](https://llm.datasette.io/en/stable/setup.html#api-key-management)
covers stored keys and environment variables.

## Load the extension

Start IPython or a Jupyter IPython kernel from the configured environment and
load the extension:

```ipython
%load_ext ipyllm
```

To load it automatically each time IPython starts, add this to the profile's
`ipython_config.py`:

```python
c.InteractiveShellApp.extensions = ["ipyllm"]
```

See the [IPython configuration guide](https://ipython.readthedocs.io/en/stable/config/intro.html)
for profile locations and configuration details.
