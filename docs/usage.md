# Usage

Use `%llm` for a single-line prompt or `%%llm` for a multi-line prompt:

```ipython
%llm What variables are available? Explain `data`.
```

```ipython
%%llm
Plot the time series in `data` with matplotlib and run the plotting code.
```

The conversation stays in memory for the current IPython session. The model
starts with variable names and Python types only, and can request additional
context through tools:

- `list_variables` lists visible names and types.
- `inspect_variable` returns a bounded representation of one variable.
- `get_history` reads recent inputs, or a requested IPython history range such
  as `1-5`.
- `execute_code` runs code in the active shell, allowing it to use live objects
  and display plots.

## Commands

| Command | Purpose |
| --- | --- |
| `%llm --model MODEL` | Select a model for this session; starts a fresh conversation. |
| `%llm --approval on` | Ask before executing code requested by the model. |
| `%llm --approval off` | Execute model-requested code without an approval prompt. |
| `%llm --config` | Show the effective model and approval setting. |
| `%llm --reset` | Clear the current conversation. |

Code execution is automatic by default. To require approval by default, add
these settings to the IPython profile's `ipython_config.py`:

```python
c.IpyllmMagics.require_approval = True
c.IpyllmMagics.model = "MODEL"
```

Session commands `%llm --approval ...` and `%llm --model ...` override the
profile settings for the current shell. The approval prompt requires the exact
response `yes`.

!!! danger "Execution is not sandboxed"
    Executed code has the same permissions as the IPython process. It can read
    or modify files, use the network, and change session objects. Approval is a
    review step, not a security boundary. Use trusted models/providers and do
    not expose secrets to a provider that should not receive them.
