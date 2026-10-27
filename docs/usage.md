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

## Diagnose a shell command

Run a command inside `%%llm --run` to capture its output and ask the model for
help understanding the result:

```ipython
%%llm --run Explain the likely cause and suggest a fix.
pytest -q
```

The command runs locally in a shell. `ipyllm` displays its stdout, stderr, and
exit status, then sends the captured output and your diagnostic request to the
current model conversation. If you omit the request, it asks for a diagnosis
and suggestions without rerunning the command:

```ipython
%%llm --run
python -m pip check
```

This is opt-in; normal REPL output is not captured or sent automatically.
Output is limited to 12,000 characters in the model prompt (keeping the
beginning and end), and commands time out after 120 seconds by default. Change
the timeout in `ipython_config.py` with
`c.IpyllmMagics.command_timeout = 60`.

Only run commands you intended to execute. Command output may contain
credentials, private paths, or other sensitive data; review it and your model
provider's data handling before sending it.

## Debug in a live script with `ipyllm.embed`

Drop a debugging session into any point of a running script with
`ipyllm.embed()`. It works like `IPython.terminal.embed()`: the shell sees
the function's live local variables, and the script resumes when you exit.
The difference is that `ipyllm` is already loaded, so you can start
prompting right away:

```python
import ipyllm

def process(rows):
    result = heavy_computation(rows)
    if result is None:
        ipyllm.embed()  # inspect `rows` and `result` interactively
    return result
```

Inside the embedded shell, `ipyllm` sees the surrounding function's local
variables:

```ipython
%llm What is holding `result` in a None state? Suggest how to fix the logic above.
```

`ipyllm.embed()` accepts the same arguments as `IPython.terminal.embed()`,
for example `header` and `config`. If you already use `IPython.embed()`,
load the extension inside the shell with `%load_ext ipyllm` instead.

Because the shell runs in the same process and frame, `ipyllm`'s tools
(`list_variables`, `inspect_variable`, `get_history`, `execute_code`) operate
directly on the local names at that point in the code. You can ask the model
to trace how an unexpected value was produced, check state of earlier
variables, or run fix-and-retry experiments on the live objects before
continuing.

The usual caveats apply: embedded code is not sandboxed, and anything you ask
the model to inspect or execute has the same permissions as the running
process.

## Use Agent Skills

`ipyllm` can load local [Agent Skills](https://agentskills.io/) as task
instructions. Each skill is a directory containing a `SKILL.md` with YAML
frontmatter (`name` and `description`) followed by Markdown instructions.

By default, skills are discovered in project- and user-level directories named
`.agents/skills/`, `.claude/skills/`, `.github/skills/`, `.cursor/skills/`,
and `.windsurf/skills/`. List available skills and descriptions with:

```ipython
%llm --skills
```

Select a skill explicitly as the first prompt token. This works for both line
and cell prompts:

```ipython
%llm /pandas Find and fix the missing values in `data`.
```

```ipython
%%llm /seaborn
Compare the distributions in `data` and explain the plot.
```

Only the selected `SKILL.md` instructions are added to that turn's prompt;
other skills are not loaded into the conversation. The skill guides the model
and does not run its scripts or grant new tools. Skills are local text and
should be reviewed like any other prompt context.

Add extra skill directories in the IPython profile's `ipython_config.py`:

```python
c.IpyllmMagics.skill_paths = [
    "~/my-skills",
    "./team/skills",
]
```

Relative paths are resolved from the current working directory. If two
discovered skills use the same name, invocation is rejected as ambiguous.
Malformed skill metadata and missing configured directories are reported by
`%llm --skills`. Skill instruction files larger than 32,000 characters are
rejected.

## Commands

| Command | Purpose |
| --- | --- |
| `%llm --model MODEL` | Select a model for this session; starts a fresh conversation. |
| `%llm --approval on` | Ask before executing code requested by the model. |
| `%llm --approval off` | Execute model-requested code without an approval prompt. |
| `%llm --skills` | List discovered skills and descriptions. |
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
