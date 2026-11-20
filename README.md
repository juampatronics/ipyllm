<div align="center">
  <img src="https://juampatronics.com/images/ipyllm_logo.svg" alt="Synaptic Prompt Logo" width="150"/>
  <br/>
  <a href="https://github.com/juampatronics/ipyllm/actions/workflows/ci.yml" alt="Build (source code)">
    <img src="https://img.shields.io/github/actions/workflow/status/juampatronics/ipyllm/ci.yml?job=build&label=build" alt="Build (source code)"/>
  </a>
  <a href="https://github.com/juampatronics/ipyllm/actions/workflows/ci.yml" alt="Docs (MkDocs)">
    <img src="https://img.shields.io/github/actions/workflow/status/juampatronics/ipyllm/ci.yml?job=docs&label=docs" alt="Docs (MkDocs)"/>
  </a>
  <a href="https://github.com/juampatronics/ipyllm/actions/workflows/ci.yml" alt="Tests (pytest)">
    <img src="https://img.shields.io/github/actions/workflow/status/juampatronics/ipyllm/ci.yml?job=test&label=tests" alt="Tests (pytest)"/>
  </a>
  <a href="https://pypi.org/project/ipyllm/" alt="PyPI version">
    <img src="https://img.shields.io/pypi/v/ipyllm?label=pypi" alt="PyPI version"/>
  </a>
</div>

# ipyllm

An IPython extension for discussing your interactive Python session with models
supported by the [`llm`](https://llm.datasette.io/) Python package. For a detailed description,
as well as a discussion of use cases, please refer to the [blog](https://juampatronics.com/tags/ipyllm/).

## Install and load

Install `ipyllm` from PyPI into the Python environment used by IPython:

```console
python -m pip install ipyllm
```

Or install from a local checkout:

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
configuration, prompts, adding files to the context, command-output and
last-error diagnosis, debugging with `ipyllm.embed`, available tools, Agent
Skills, per-prompt and per-model token usage,
approval settings, and safety. Preview or build the site with:

```console
uv run mkdocs serve
uv run mkdocs build
```

An online version is available at the [homepage](https://juampatronics.com/ipyllm/).

