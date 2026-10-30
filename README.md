<div align="center">
  <img src="docs/ipyllm_logo.svg" alt="Synaptic Prompt Logo" width="150"/>
  <br/>
  <a href="https://github.com/juampatronics/ipyllm/actions/workflows/ci.yml?query=job%3Abuild" alt="Build">
    <img src="https://github.com/juampatronics/ipyllm/actions/workflows/ci.yml/badge.svg?job=build" alt="Build"/>
  </a>
  <a href="https://github.com/juampatronics/ipyllm/actions/workflows/ci.yml?query=job%3Adocs" alt="Docs">
    <img src="https://github.com/juampatronics/ipyllm/actions/workflows/ci.yml/badge.svg?job=docs" alt="Docs"/>
  </a>
  <a href="https://github.com/juampatronics/ipyllm/actions/workflows/ci.yml?query=job%3Atest" alt="Tests">
    <img src="https://github.com/juampatronics/ipyllm/actions/workflows/ci.yml/badge.svg?job=test" alt="Tests"/>
  </a>
  <a href="https://pypi.org/project/ipyllm/" alt="PyPI version">
    <img src="https://img.shields.io/pypi/v/ipyllm?label=pypi" alt="PyPI version"/>
  </a>
</div>

# ipyllm

An IPython extension for discussing your interactive Python session with models
supported by the [`llm`](https://llm.datasette.io/) Python package.

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
configuration, prompts, command-output diagnosis, debugging with
`ipyllm.embed`, available tools, Agent Skills, approval settings, and
safety. Preview or build the site with:

```console
uv run mkdocs serve
uv run mkdocs build
```
