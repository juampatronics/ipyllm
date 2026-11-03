# Usage

Use `%llm` for a single-line prompt or `%%llm` for a multi-line prompt:

```ipython
%llm What variables are available? Explain `data`.
```

```ipython
%%llm
Plot the time series in `data` with matplotlib and run the plotting code.
```

!!! note "Prompts ending in `?`"
    IPython treats a line that ends in `?` as a request for object help
    (`foo?` → inspect `foo`), and rewrites it before the magic runs. A
    line-magic prompt that ends in `?` therefore never reaches the model:

    ```ipython
    %llm What do you think of this plan?   # IPython turns this into `pinfo plan`
    ```

    Two ways to send a question that ends in `?`: use the cell magic (the
    body is passed through unchanged),

    ```ipython
    %%llm
    What do you think of this plan?
    ```

    or end the single-line prompt without the trailing `?`. A `?` anywhere
    else in the prompt is fine.

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

## Diagnose the last error

When a cell raises an exception you don't need to rerun it for help.
`%llm --last-error` captures the most recent traceback and asks the model to
diagnose it:

```ipython
def divide(a, b):
    return a / b

divide(1, 0)            # ZeroDivisionError
%llm --last-error explain why this failed and how to fix it
```

`ipyllm` displays the captured traceback, then sends it and your request to the
current model conversation. Omit the request to ask for a diagnosis and a fix
suggestion. The traceback is rendered the way IPython displays it (including
the source line where the error occurred), with ANSI color codes removed, and
is limited to 12,000 characters in the model prompt (keeping the beginning and
end). Only the most recent exception is captured: if a later cell fails, or the
model itself runs code that fails, that error becomes the one diagnosed.

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

## Add files to the context

Reference local files directly in a prompt with `@path`. When you type `@`
at the start of a word, IPython autocompletes the names of files and
directories in the referenced folder (directories are shown with a trailing
`/`, so you can tab into `@sub/` and keep going). You can reference several
files in one prompt:

```ipython
%llm @data.py @notes.md Explain how `data.py` builds the table described in `notes.md`.
```

```ipython
%llm @plots/ Describe the trend across these plots.
```

How a referenced file reaches the model depends on its type:

- **Text files** (code, Markdown, CSV, JSON, plain text, …) contribute their
  content, labeled with the file name, so the model reads them inline. Each
  file's content is limited to 12,000 characters.
- **Media files** (images, audio, video, and PDF) are sent as
  `llm` attachments. They must be a type your model supports; otherwise `llm`
  reports the unsupported type.
- **Directories** are expanded to the regular files they contain (searched
  breadth-first, hidden entries skipped, at most 100 files).

Paths are resolved relative to the current working directory; `~` and
absolute paths work too. A reference that does not resolve to an existing
file or directory is left in the prompt as literal text, so email addresses
and other in-word `@` are unaffected. Repeated references are sent only once.
Binary files that are neither text nor a supported media type are rejected
with a clear error.

!!! note "File contents are sent to the provider"
    Referenced files are read locally and their content (or the file itself,
    for media) is included in the request to the model. Review the files and
    the provider's data handling before referencing sensitive ones.

## Commands

| Command | Purpose |
| --- | --- |
| `%llm --model MODEL` | Select a model for this session; starts a fresh conversation. |
| `%llm --approval on` | Ask before executing code requested by the model. |
| `%llm --approval off` | Execute model-requested code without an approval prompt. |
| `%llm --debug on` | Show full IPython tracebacks for errors in model-executed code. |
| `%llm --debug off` | Hide those tracebacks (the default). |
| `%llm --skills` | List discovered skills and descriptions. |
| `%llm --save FILE.py` | Save subsequent requests as comments and model-executed Python to a new script. |
| `%llm --save off` | Stop saving to the current script. |
| `%llm --resume FILE.py` | Execute a saved session in the current namespace and continue appending to it. |
| `%llm --chain-limit N` | Set the maximum model/tool rounds per prompt (default: `10`). |
| `%llm --config` | Show the effective model and approval setting. |
| `%llm --reset` | Clear the current conversation. |

Raise the limit for prompts where the model needs several tool interactions:

```ipython
%llm --chain-limit 25
%llm Inspect the data, create a plot, and summarize the results.
```

The value must be a positive integer. You can also set a default in the
IPython profile's `ipython_config.py`:

```python
c.IpyllmMagics.chain_limit = 25
```

Each prompt uses the current limit. A larger value allows more model/tool
rounds, but can increase latency and model usage.

To record a session, enable saving before making requests:

```ipython
%llm --save session_file.py
%llm Create a small example DataFrame and summarize it.
```

Each request is written as Python comments, followed by code executed through
the model's `execute_code` tool. The file is created in the current working
directory (or at the path you provide); existing files are never overwritten.
Saving starts when `--save` is issued, so earlier requests, manually-entered
code, and variables created before recording are not included. The script may
need setup code or data-loading steps to run independently. IPython-only input
is preserved as comments because it is not valid standalone Python.

To continue a saved session in a new IPython process, replay its Python code
and resume appending:

```ipython
%llm --resume session_file.py
%llm Continue analyzing the data.
```

`--resume` executes the file in the current IPython namespace and appends
subsequent requests and model-executed code to that same file. It extracts
saved request comments as background for the next LLM prompt, but does not
restore the prior model responses or full conversation. Resuming requires a
valid `.py` file and is unavailable while another save target is active; use
`%llm --save off` first. The session script is executable Python and may repeat
side effects such as writing files or making network requests, so inspect it
before resuming. Manually entered code and pre-existing data are not included
unless they were captured in the file; IPython-only commands remain comments
and do not run during replay.

Code execution is automatic by default. To require approval by default, add
these settings to the IPython profile's `ipython_config.py`:

```python
c.IpyllmMagics.require_approval = True
c.IpyllmMagics.model = "MODEL"
```

Errors from model-executed code are returned to the model so it can diagnose
and retry, but their tracebacks are hidden from the REPL by default to avoid
cluttering iterative work. Enable them when debugging the extension or the
generated code:

```ipython
%llm --debug on
```

Use `%llm --debug off` to hide them again. To show tracebacks by default, set
`c.IpyllmMagics.debug = True` in `ipython_config.py`. This setting does not
suppress successful output or plots.

Session commands `%llm --approval ...`, `%llm --debug ...`, and
`%llm --model ...` override the applicable profile settings for the current
shell. The approval prompt requires the exact response `yes`.

!!! danger "Execution is not sandboxed"
    Executed code has the same permissions as the IPython process. It can read
    or modify files, use the network, and change session objects. Approval is a
    review step, not a security boundary. Use trusted models/providers and do
    not expose secrets to a provider that should not receive them.
