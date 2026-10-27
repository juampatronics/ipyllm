<div align="center">
  <img src="docs/logos/ipyllm_logo.svg" alt="Synaptic Prompt Logo" width="150"/>
</div>

# ipyllm

An IPython extension for discussing your interactive Python session with models
supported by the [`llm`](https://llm.datasette.io/) Python package.

## Install and load

Install the package in the Python environment used by IPython:

```console
python -m pip install .
```

Then load it in IPython or a Jupyter IPython kernel:

```ipython
%load_ext ipyllm
```

The extension uses the model and API-key configuration already managed by
`llm`. For example, choose a default model with `llm models default MODEL` and
configure provider keys using `llm keys set PROVIDER`.

## Documentation

The [MkDocs documentation](docs/index.md) covers installation, model
configuration, prompts, command-output diagnosis, debugging with
`ipyllm.embed`, available tools, Agent Skills, approval settings, and
safety. Preview or build the site with:

```console
uv run mkdocs serve
uv run mkdocs build
```
