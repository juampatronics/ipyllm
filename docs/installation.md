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
