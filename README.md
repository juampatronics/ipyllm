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

## Use

Ask a question on a line or in a cell:

```ipython
%llm Explain what `data` contains. Inspect it if useful.

%%llm
Plot the time series in `data` using matplotlib and execute the plotting code.
```

The model can request these tools during the conversation:

- `list_variables` returns visible variable names and Python types.
- `inspect_variable` returns a bounded `repr()` for one requested variable.
- `get_history` reads a bounded set of recent inputs, or an explicit IPython
  history range such as `1-5`.
- `execute_code` runs Python in the active IPython session, so the model can
  work with live objects and display plots in that session.

Variable values and history are not sent with every prompt; the model must
request them as needed. The conversation is kept in memory for the current
IPython session. `%llm --reset` starts a fresh conversation.

Select a model for the current session with `%llm --model MODEL`. Toggle an
approval prompt before model-requested code execution with
`%llm --approval on` or `%llm --approval off`. `%llm --config` shows the
effective model and approval setting. The approval prompt requires the exact
response `yes`.

The default setting executes model-requested code without prompting. To require
approval by default in an IPython profile, add:

```python
c.IpyllmMagics.require_approval = True
c.IpyllmMagics.model = "MODEL"
```

to that profile's `ipython_config.py`. Runtime `%llm --approval ...` and
`%llm --model ...` settings override the profile for the current shell.

## Safety and privacy

Code execution happens in the live Python process with the same permissions as
the user. It is not sandboxed: code can read or modify files, access the
network, and change or delete objects in the session. Approval is a review
guardrail, not a security boundary. Use trusted models and providers, and
enable approval when reviewing generated code before it runs.

Only variable names/types are initially provided, but the model may request
specific variable representations or history, and may ask to execute code.
Avoid placing secrets in the session when using a model/provider that should
not receive them.
