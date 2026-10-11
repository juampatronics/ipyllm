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
