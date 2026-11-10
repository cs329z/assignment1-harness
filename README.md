![Agent Harness: CS329Z Assignment 1](assets/agent-harness.png)

# CS329Z Assignment 1: Building an Agent Harness

This is the starter repository for Assignment 1.

**Start by reading the handout, [`cs329z_assignment1_harness.pdf`](cs329z_assignment1_harness.pdf), in this directory.**
It states every problem, what is graded, and the design decisions that
are yours to make. This file covers setup, the commands you will run, and
where things are.


## Setup

1. Install `uv`, which manages Python and the dependencies:
   https://docs.astral.sh/uv/getting-started/installation/

   The assignment needs Python 3.11 or newer. `uv` installs one if you do
   not have it. macOS, Linux and Windows are supported. On Windows, also
   install Git for Windows (https://git-scm.com/download/win, default
   options): the terminal sandbox runs commands with the `bash.exe` it
   ships, and there is no operating-system jail on Windows (see "The
   terminal sandbox" below). WSL works too and gives you the Linux jail.

2. Create your `.env` file and add your API key:

   ```sh
   cp .env.example .env
   ```

   Open `.env` and fill in `OPENROUTER_API_KEY`. The file is gitignored. Do not
   commit it.


3. Unpack the email archive. It ships in the repository as
   `data/emails.jsonl.gz` (19 MB). This command verifies its checksum and
   unpacks it to `data/emails/emails.jsonl` (58 MB):

   ```sh
   uv run python data/download.py
   ```

4. Run the tests. In the starter, every test fails with `NotImplementedError`:

   ```sh
   uv run pytest
   ```

Run every command from the repository root with `uv run`. The first
`uv run` creates the environment in `.venv/`.

## Tests

There are two kinds of tests, in the same files under `tests/`.

**Deterministic tests** run by default and make no model calls. Where your
code needs a model, they pass a `ScriptedLM`, a stand-in that replays a fixed
list of replies.

```sh
uv run pytest                          # all deterministic tests
uv run pytest tests/test_priority.py   # one problem
```

**Live tests** call the real model and compare your results against the
thresholds in `tests/thresholds.py`. They cost money.

```sh
uv run pytest -m live -s                          # all live tests
uv run pytest tests/test_priority.py -m live -s   # one problem
```

A live test takes from a few seconds to a few minutes (`priority` makes 60
model calls, about a minute), and pytest prints nothing while it waits.
`-s` makes each test print its score and the items your code got wrong,
whether or not it passes.

While you iterate, `CS329Z_EVAL_SLICE=N` runs each live test on its first
N items only. You can set it on the command line or in `.env`:

```sh
CS329Z_EVAL_SLICE=5 uv run pytest tests/test_priority.py -m live -s
```

## Seeing what your pipelines produce

The tests report scores. To read the output itself, run a pipeline on real
data:

```sh
uv run python -m cs329z_hw1.show priority em-12167              # label and reason for one email
uv run python -m cs329z_hw1.show digest 2001-06-22 --labels --limit 30   # the digest, plus each label
uv run python -m cs329z_hw1.show bm25 "larchfield audit"                 # top emails for a query, with scores
uv run python -m cs329z_hw1.show email_qa "Who leads the Basin Analytics move?"
uv run python -m cs329z_hw1.show search_docs "parental leave"
uv run python -m cs329z_hw1.show memory .lm_cache/chat_memory             # what chat.py's agent has stored
uv run python -m cs329z_hw1.show memory                                   # each persona's memory after an evaluation run
```

Each prints the result and the cost and time of the model calls. A repeat
with unchanged code is free.

## Model calls and spending

All model calls go through `cs329z_hw1.llm.LM`: a list of
`{"role", "content"}` messages in, a string out.

```python
from cs329z_hw1.llm import LM

lm = LM()            # the grading model, openai/gpt-6-luna
judge = LM("judge")  # the evaluation's judge model, openai/gpt-6-sol
text = lm([{"role": "user", "content": "Say hello."}])
```

The wrapper does three things:

- **Cache.** Each reply is stored under `.lm_cache/`, keyed on the model,
  the sampling settings and the exact messages. Repeating an identical call
  returns the stored reply and costs nothing.
- **Ledger.** Each call that reaches the provider is appended to
  `.lm_cache/ledger.jsonl` with its token counts and cost. Print the totals
  by model and by tag (the test suite tags each call with the test name)
  with:

  ```sh
  uv run python -m cs329z_hw1.llm
  ```

- **Budget.** The key we issue you has a spending limit on our side, and
  calls fail once it is reached. `CS329Z_BUDGET_USD` (in `.env` or the
  environment) is a second limit you set for yourself: once the ledger
  total reaches it, the wrapper raises `BudgetExceeded` instead of making
  a call, so a runaway loop stops before the key does. Cached replies are
  still returned.

Deleting `.lm_cache/` deletes both the cache and the ledger.

## How to work a problem

Your code goes anywhere inside the `cs329z_hw1` package, and the tests
reach it only through the functions in `cs329z_hw1/adapters.py`, which ship
as stubs that raise `NotImplementedError`.

For each problem:

1. Read the problem in the handout.
2. Open the adapter it names in `cs329z_hw1/adapters.py`. The adapter's type
   hints and docstring are the exact signature the tests use.
3. Implement the functionality in your package, for example in
   `cs329z_hw1/pipelines/priority.py`.
4. Replace the stub's body with a call into your code:

   ```python
   def run_priority(email, lm):
       from cs329z_hw1.pipelines.priority import classify_priority
       return classify_priority(email, lm)
   ```

5. Run that problem's tests, deterministic first and then live.

The deterministic tests are the complete list of the error handling we
require (an empty reply, a reply in the wrong format, a cut-off tool call,
a tool that raises). Nothing is graded that is not in a test or asked for
in the design memo.

The types that cross the adapter boundary (`Email`, `ToolSpec`, `ToolCall`,
`ToolResult`, `ParsedResponse`, `AgentConfig`, `AgentResult`, and others) are
defined in `cs329z_hw1/types.py`. Use them as they are.

## Repository layout

Files marked *provided* are course code, which grading replaces with its
own copies, so read them but do not edit them. Files marked *read* define
interfaces you build on or rules the tests check, and each is worth
reading before the problem that first uses it.

```
README.md  pyproject.toml  .env.example
cs329z_hw1/                               your package: every file you write goes here
  adapters.py           yours     read  the only file the tests call; starts as stubs
  pipelines/            yours           Part 1 (starts empty)
  agent/                yours           Part 2 (starts empty)
  llm.py                provided  read  LM, ScriptedLM, cache, ledger, budget
  types.py              provided  read  data types shared by your code and the tests
  tokens.py             provided  read  count_tokens, count_message_tokens (4 characters per token)
  tokenizer.py          provided        tokenize(), the tokenizer BM25 must use
  data.py               provided  read  load_emails(), emails_on(), email_text(), load_docs()
  sandbox.py            provided  read  run_terminal(), make_workspace()
  cardinal.py           provided  read  names and argument schemas of the agent tools
  user.py               provided  read  UserIO, ScriptedUser, ConsoleUser
  chat.py               provided        terminal chat with your agent
  show.py               provided        run one Part 1 pipeline on real data and print the result
  simulation/           provided        simulated users, judge, evaluation runner (see its README.md)
tests/                                    provided: read, do not edit; discarded at grading
  conftest.py           provided
  helpers.py            provided
  thresholds.py         provided        pass thresholds for the live tests
  test_*.py             provided  read  one file per problem; read the one you are on
  fixtures/             provided        gold labels, scripted sessions, personas
data/
  docs/                 provided        Cardinal Energy documents
  emails.jsonl.gz       provided        the email archive, compressed
  download.py           provided        verifies and unpacks the archive
  source.json           provided        checksum of the archive
  priority_rubric.md    provided  read  how the gold priority labels were assigned
  emails/                         unpacked by download.py, gitignored
```


You may add files and subpackages inside `cs329z_hw1/`. Grading copies
your whole `cs329z_hw1/` package into a clean starter and discards
everything else, so a helper you put under `tests/` or anywhere outside the
package is not there when we grade. Do not add dependencies to
`pyproject.toml`; NumPy is already one, and you may use it anywhere.

## The terminal sandbox

`cs329z_hw1.sandbox.run_terminal(cmd)` runs one shell command for your
agent. Read `cs329z_hw1/sandbox.py` before you use it. Its module docstring
lists what each layer stops and what it does not. One of those layers is an
operating-system jail that is not available on every machine. To see which
one your machine uses:

```sh
uv run python -c "from cs329z_hw1 import sandbox; print(sandbox.jail())"
```

This prints `sandbox-exec` (macOS), `bwrap` (Linux with bubblewrap
installed and permitted), or `none`. With `none`, a command that runs
`python3` can reach the network and any file your account can. On Linux you
can install bubblewrap with your package manager (for example
`sudo apt install bubblewrap`).

On Windows it always prints `none`: there is no OS jail there, so the
allowlist, the scrubbed environment and the timeout are the only limits.
The shell and the allowed programs (`grep`, `sort`, `awk` and the rest)
come from Git for Windows, so it must be installed; without it every
command is rejected with a message that says so. Commands are written as on
Linux (`grep`, `wc -l`, pipes, `python3 -c '...'`); `python3` is the Python
that `uv` installed, without the project's packages. Line endings in a
command's output are normalized to `\n`.

Each command runs in a workspace directory that holds `emails.jsonl` and
`docs/`. Here-documents (`<<EOF`) are rejected; multi-line Python goes in
`python3 -c '...'`.

## The simulated-user evaluation

```sh
uv run python -m cs329z_hw1.simulation --slice 3   # the first 3 personas
uv run python -m cs329z_hw1.simulation             # all public personas
```

This plays each persona in `tests/fixtures/personas/` against the agent
behind `run_agent_session` and saves the conversations under `runs/`. Pass a
directory as the first argument to use other personas. See
`uv run python -m cs329z_hw1.simulation --help` for the options.

## Chatting with your agent

Once `run_agent_session` works, you can talk to your agent in the terminal.

```sh
uv run python -m cs329z_hw1.chat [--mode confirm|auto] [--memory-dir DIR]
```

## Acknowledgements

This assignment was designed and built by Michael Ryan for Stanford's
CS329Z Engineering AI Agents course with lots of help from Claude Fable 5.1.
We owe a debt of gratitude to the various people who shared thoughts and
feedback on this assignment. In particular we would like to highlight Houjun
(Jack) Liu, Owen Queen, Shreyas Sharma, and Anantharaman Iyer for completing a
preliminary version of the assignment and offering invaluable feedback.

If you would like to use this assignment for your course you are welcome to
do so with brief attribution. It is released under the MIT license (see
`LICENSE`).
