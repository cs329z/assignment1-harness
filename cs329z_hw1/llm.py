"""Course-provided language model wrapper.

This module gives you exactly one primitive: a chat-completion call that takes
a list of messages and returns a string.

    lm = LM()                                  # the grading model
    text = lm([{"role": "system", "content": "..."},
               {"role": "user", "content": "..."}])

There is no tool-calling parameter and no structured-output parameter. Those
are yours to build on top of this call.

What the wrapper does for you:

* Disk cache. Every call is cached under ``.lm_cache/`` keyed on the model,
  the sampling settings, and the exact messages. Re-running unchanged code
  costs nothing.
* Cost ledger. Every call that reaches the provider is appended to
  ``.lm_cache/ledger.jsonl`` with its token counts and dollar cost.
  ``uv run python -m cs329z_hw1.llm`` prints a spending report.
* Budget stop. If ``CS329Z_BUDGET_USD`` is set, the wrapper raises
  ``BudgetExceeded`` instead of making a call once the ledger total passes it.
* Truncation. A call that stops because the model reached ``max_tokens``
  raises ``OutputTruncated`` instead of returning the cut-off text. The
  exception carries the text produced so far (``.text``) and the token
  counts. The call is charged and recorded in the ledger but not cached.

``ScriptedLM`` is a drop-in fake used by the deterministic tests. It replays a
fixed list of replies and never touches the network.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

Message = dict  # {"role": "system" | "user" | "assistant", "content": str}

# Model used for each role. Override any of these with the environment
# variable CS329Z_<ROLE>_MODEL, e.g. CS329Z_GRADING_MODEL.
DEFAULT_MODELS = {
    "grading": "openai/gpt-6-luna",  # the model your agent is graded with
    "user": "openai/gpt-6-luna",  # plays the simulated users
    "judge": "openai/gpt-6-sol",  # scores evaluation transcripts
}

# Calls go to OpenRouter, which fronts many providers behind one
# OpenAI-compatible API. Model names carry the provider as a prefix
# ("openai/gpt-6-luna"); https://openrouter.ai/models lists them.
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# Local addition, not part of the course starter: DeepSeek as a second
# provider, for working without an OpenRouter key. It is used when
# CS329Z_PROVIDER=deepseek, or when CS329Z_PROVIDER is unset and
# DEEPSEEK_API_KEY is set. Grading replaces this file and uses the models
# above, so live-test results on DeepSeek are an approximation of the grade.
# https://api-docs.deepseek.com lists the models.
DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODELS = {
    "grading": "deepseek-flash",
    "user": "deepseek-flash",
    "judge": "deepseek-flash",  # CS329Z_JUDGE_MODEL=deepseek-v4-pro for the other model
}
PROVIDERS = ("openrouter", "deepseek")

# Reasoning effort requested from models that accept it (GPT-5 and GPT-6).
# Override with CS329Z_REASONING_EFFORT.
DEFAULT_REASONING_EFFORT = "low"

# USD per 1M tokens: (input, cached input, output), keyed by the model name
# without its provider prefix. OpenRouter reports the exact cost of each
# call in the response, and the ledger records that; this table is the
# fallback for a response without one, and it is checked when an LM is
# created so that an unknown model fails before a paid call. Read from
# https://openrouter.ai/models on 2026-10-05 (the same as OpenAI's list
# prices).
PRICES = {
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-6-astra": (10.00, 1.00, 50.00),
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6-sol": (2.00, 0.20, 10.00),
    # DeepSeek does not report a cost, so these are what the ledger uses.
    # They are the peak-hour prices (off-peak is half), so the ledger is an
    # upper bound. Read from https://api-docs.deepseek.com/quick_start/pricing
    # on 2026-11-13.
    "deepseek-flash": (0.30, 0.006, 1.20),
    "deepseek-v4-pro": (1.32, 0.044, 3.96),
}

VALID_ROLES = ("system", "user", "assistant")

# Finish reasons with which DeepSeek reports a reply it did not finish for a
# reason of its own (not max_tokens). Such a reply is not returned or cached.
_INTERRUPTED = ("insufficient_system_resource", "aborted")


class BudgetExceeded(RuntimeError):
    """Raised instead of calling the provider once the spend cap is reached."""


class ScriptExhausted(RuntimeError):
    """Raised when a ScriptedLM is called more times than its script allows."""


class OutputTruncated(RuntimeError):
    """Raised when the model stopped because it reached ``max_tokens``.

    ``text`` is what the model wrote before the cut, which is often empty:
    a reasoning model's hidden reasoning counts against the same limit, so
    the limit can be used up before any visible text is written.
    ``output_tokens`` is the whole output, ``reasoning_tokens`` the hidden
    part of it (0 when the provider does not report it). Calling again with
    a larger ``max_tokens`` is a new paid call.
    """

    def __init__(self, text: str, *, model: str, max_tokens: int, output_tokens: int, reasoning_tokens: int = 0):
        self.text = text
        self.model = model
        self.max_tokens = max_tokens
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens
        shown = f"{len(text)} characters of text" if text else "no visible text"
        super().__init__(
            f"{model} stopped at max_tokens={max_tokens} with {shown} "
            f"({output_tokens} output tokens, {reasoning_tokens} of them hidden reasoning). "
            "Ask for a shorter reply, or call again with a larger max_tokens."
        )


def resolve_model(role_or_model: str | None = None) -> str:
    """Map a role ("grading", "user", "judge") to a model name.

    ``None`` means "grading". A string that is not a role is returned
    unchanged, so you can pass a model name.
    """
    name = role_or_model or "grading"
    if name in DEFAULT_MODELS:
        defaults = DEEPSEEK_MODELS if provider() == "deepseek" else DEFAULT_MODELS
        return os.environ.get(f"CS329Z_{name.upper()}_MODEL", defaults[name])
    return name


def provider() -> str:
    """Which API the calls go to: "openrouter" or "deepseek".

    ``CS329Z_PROVIDER`` decides. Without it, DeepSeek is used if
    ``DEEPSEEK_API_KEY`` is set, and OpenRouter otherwise.
    """
    _load_env()  # the keys are usually in .env
    name = os.environ.get("CS329Z_PROVIDER", "").strip().lower()
    if not name:
        name = "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else "openrouter"
    if name not in PROVIDERS:
        raise ValueError(f"CS329Z_PROVIDER must be one of {PROVIDERS}, got {name!r}")
    return name


def _price(model: str) -> tuple[float, float, float]:
    model = model.rsplit("/", 1)[-1]  # "openai/gpt-6-luna" -> "gpt-6-luna"
    best = None
    for prefix, price in PRICES.items():
        if model == prefix or model.startswith(prefix + "-20"):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, price)
    if best is None:
        raise KeyError(
            f"No price on file for model {model!r}. Add it to PRICES in cs329z_hw1/llm.py."
        )
    return best[1]


def cost_usd(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    p_in, p_cached, p_out = _price(model)
    fresh = max(input_tokens - cached_tokens, 0)
    return (fresh * p_in + cached_tokens * p_cached + output_tokens * p_out) / 1_000_000


def _cache_dir() -> Path:
    return Path(os.environ.get("CS329Z_CACHE_DIR", REPO_ROOT / ".lm_cache"))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(REPO_ROOT / ".env")


_ledger_lock = threading.Lock()
# Models that reject the temperature or reasoning_effort parameters, learned
# from the provider's error the first time each is used.
_NO_TEMPERATURE: set[str] = set()
_NO_EFFORT: set[str] = set()


def read_ledger() -> list[dict]:
    path = _cache_dir() / "ledger.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def total_spend() -> float:
    return sum(row.get("cost_usd", 0.0) for row in read_ledger())


def _append_ledger(row: dict) -> None:
    path = _cache_dir() / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ledger_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def _validate_messages(messages: Sequence[Message]) -> None:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a non-empty list of {'role', 'content'} dicts")
    for m in messages:
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            raise ValueError(
                f"each message must be a dict with exactly the keys 'role' and 'content', got {m!r}"
            )
        if m["role"] not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {m['role']!r}")
        if not isinstance(m["content"], str):
            raise ValueError("message content must be a string")


class LM:
    """A chat model: messages in, string out.

    Parameters
    ----------
    model:
        A role ("grading", "user", "judge"), an OpenRouter model name such
        as "openai/gpt-6-sol" (a DeepSeek model name such as
        "deepseek-v4-pro" when the provider is DeepSeek, see ``provider``),
        or ``None`` for "grading".
    temperature, max_tokens:
        Sampling settings. They are part of the cache key. Reasoning models
        spend part of ``max_tokens`` on hidden reasoning, so keep it large.
        Models that reject ``temperature`` are called without it.
    tag:
        A free-form label written to the cost ledger so spending can be
        grouped (the test suite sets it to the test name).
    cache:
        Set to False to bypass the disk cache for this instance.
    salt:
        Extra string mixed into the cache key. Two instances with different
        salts get independent samples for identical messages.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tag: str | None = None,
        cache: bool = True,
        salt: str = "",
    ) -> None:
        _load_env()
        self.model = resolve_model(model)
        _price(self.model)  # fail now, not after a paid call, if the price is unknown
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tag = tag
        self.cache = cache
        self.salt = salt
        # Running totals for this instance.
        self.calls = 0
        self.cached_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.seconds = 0.0
        self._client = None
        self._provider = provider()

    # -- public API -------------------------------------------------------

    def __call__(self, messages: Sequence[Message]) -> str:
        _validate_messages(messages)
        messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        key = self._key(messages)
        path = _cache_dir() / key[:2] / f"{key}.json"
        self.calls += 1
        if self.cache and path.exists():
            self.cached_calls += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]

        self._check_budget()
        start = time.time()
        text, usage = self._complete(messages)
        elapsed = time.time() - start
        cost = usage.get("cost")
        if cost is None:
            cost = cost_usd(self.model, usage["input"], usage["cached"], usage["output"])
        self.input_tokens += usage["input"]
        self.output_tokens += usage["output"]
        self.cost_usd += cost
        self.seconds += elapsed
        truncated = usage.get("finish_reason") == "length"
        _append_ledger(
            {
                "ts": round(start, 3),
                "model": self.model,
                "tag": self.tag or os.environ.get("CS329Z_TAG", ""),
                "input_tokens": usage["input"],
                "cached_tokens": usage["cached"],
                "output_tokens": usage["output"],
                "cost_usd": cost,
                "seconds": round(elapsed, 3),
                "finish_reason": usage.get("finish_reason"),
            }
        )
        if truncated:  # charged and logged, never cached
            raise OutputTruncated(
                text,
                model=self.model,
                max_tokens=self.max_tokens,
                output_tokens=usage["output"],
                reasoning_tokens=usage.get("reasoning", 0),
            )
        if usage.get("finish_reason") in _INTERRUPTED:  # charged and logged, never cached
            raise RuntimeError(
                f"{self.model} did not finish its reply (finish_reason={usage['finish_reason']!r}): "
                "the provider interrupted it. Run again."
            )
        if self.cache and text.strip():  # an empty reply is not worth remembering
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"text": text, "model": self.model}), encoding="utf-8")
            tmp.replace(path)
        return text

    def usage(self) -> dict:
        """Totals for this instance since it was created."""
        return {
            "model": self.model,
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "seconds": self.seconds,
        }

    # -- internals --------------------------------------------------------

    def _key(self, messages: list[Message]) -> str:
        effort = os.environ.get("CS329Z_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
        blob = json.dumps(
            [self.model, self.temperature, self.max_tokens, effort, self.salt, messages],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _check_budget(self) -> None:
        cap = os.environ.get("CS329Z_BUDGET_USD")
        if not cap:
            return
        spent = total_spend()
        if spent >= float(cap):
            raise BudgetExceeded(
                f"Spent ${spent:.2f} of the ${float(cap):.2f} cap set by CS329Z_BUDGET_USD. "
                "Raise the cap in .env if you mean to continue."
            )

    def _complete(self, messages: list[Message]) -> tuple[str, dict]:
        deepseek = self._provider == "deepseek"
        if self._client is None:
            from openai import OpenAI

            key_name = "DEEPSEEK_API_KEY" if deepseek else "OPENROUTER_API_KEY"
            key = os.environ.get(key_name)
            if not key:
                raise RuntimeError(f"{key_name} is not set. Copy .env.example to .env and add your key.")
            self._client = OpenAI(
                base_url=DEEPSEEK_URL if deepseek else OPENROUTER_URL,
                api_key=key,
                max_retries=6,
                timeout=120,
                default_headers={} if deepseek else {"X-Title": "CS329Z HW1"},
            )
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            # OpenRouter extensions: the reasoning effort, and the request
            # that the response carry the cost of the call. DeepSeek takes
            # the effort as a plain string and reports no cost.
            "extra_body": {} if deepseek else {"usage": {"include": True}},
        }
        if self.model not in _NO_TEMPERATURE:
            kwargs["temperature"] = self.temperature
        effort = os.environ.get("CS329Z_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
        effort_key = "reasoning_effort" if deepseek else "reasoning"
        if effort and self.model not in _NO_EFFORT:
            kwargs["extra_body"][effort_key] = effort if deepseek else {"effort": effort}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # some models reject temperature or reasoning effort
            text = str(exc)
            if "temperature" in text and "temperature" in kwargs:
                _NO_TEMPERATURE.add(self.model)
                kwargs.pop("temperature")
                response = self._client.chat.completions.create(**kwargs)
            elif "reasoning" in text and effort_key in kwargs["extra_body"]:
                _NO_EFFORT.add(self.model)
                kwargs["extra_body"].pop(effort_key)
                response = self._client.chat.completions.create(**kwargs)
            else:
                raise
        choice = response.choices[0]
        text = choice.message.content or ""
        u = response.usage
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        out_details = getattr(u, "completion_tokens_details", None)
        reasoning = getattr(out_details, "reasoning_tokens", 0) or 0
        usage = {
            "input": u.prompt_tokens,
            "cached": cached,
            "output": u.completion_tokens,
            "reasoning": reasoning,
            "finish_reason": getattr(choice, "finish_reason", None),
        }
        cost = getattr(u, "cost", None)  # OpenRouter's figure for this call
        if isinstance(cost, (int, float)):
            usage["cost"] = float(cost)
        return text, usage


class ScriptedLM:
    """A fake LM that replays a fixed script. Used by the deterministic tests.

    Each script entry is one model reply, given as one of:

    * a string, returned as is;
    * a dict ``{"text": str, "tool_call": {"name": str, "args": dict}}``,
      rendered to a string by ``formatter(text, ToolCall)``. The tests pass
      your ``run_format_tool_call`` adapter as the formatter, so scripted
      tool calls arrive in your own format;
    * a callable ``f(messages) -> str``.

    Every call's messages are recorded in ``.calls`` so tests can inspect
    exactly what your code sent to the model. When the script runs out the
    LM raises ``ScriptExhausted``, unless ``repeat_last=True``, in which case
    the last entry is replayed forever.
    """

    model = "scripted"

    def __init__(
        self,
        script: Sequence[Any],
        *,
        formatter: Callable[[str, Any], str] | None = None,
        repeat_last: bool = False,
    ) -> None:
        self.script = list(script)
        self.formatter = formatter
        self.repeat_last = repeat_last
        self.calls: list[list[Message]] = []
        self.replies: list[str] = []

    def __call__(self, messages: Sequence[Message]) -> str:
        _validate_messages(messages)
        index = len(self.calls)
        self.calls.append(copy.deepcopy(list(messages)))
        if index >= len(self.script):
            if not (self.repeat_last and self.script):
                raise ScriptExhausted(
                    f"ScriptedLM was called {index + 1} times but the script has "
                    f"{len(self.script)} entries. Your code made a model call the test "
                    "did not expect."
                )
            index = len(self.script) - 1
        entry = self.script[index]
        if callable(entry):
            reply = entry(self.calls[-1])
        elif isinstance(entry, dict):
            reply = self._render(entry)
        else:
            reply = str(entry)
        self.replies.append(reply)
        return reply

    @property
    def remaining(self) -> int:
        return max(len(self.script) - len(self.calls), 0)

    def _render(self, entry: dict) -> str:
        text = entry.get("text", "")
        call = entry.get("tool_call")
        if call is None:
            return text
        if self.formatter is None:
            raise RuntimeError("ScriptedLM needs a formatter to render scripted tool calls")
        from cs329z_hw1.types import ToolCall

        return self.formatter(text, ToolCall(name=call["name"], args=dict(call.get("args", {}))))

    def usage(self) -> dict:
        return {"model": "scripted", "calls": len(self.calls), "cost_usd": 0.0}


def spending_report(rows: list[dict] | None = None) -> str:
    rows = read_ledger() if rows is None else rows
    if not rows:
        return "No LLM calls recorded yet."
    by_model: dict[str, dict] = {}
    by_tag: dict[str, dict] = {}
    for row in rows:
        model = row.get("model", "?")
        short = model.split("-20")[0]  # drop the snapshot date
        tag_key = f"{row.get('tag') or '(untagged)'} [{short}]"
        for table, key in ((by_model, model), (by_tag, tag_key)):
            agg = table.setdefault(key, {"calls": 0, "cut": 0, "in": 0, "out": 0, "usd": 0.0, "sec": 0.0})
            agg["calls"] += 1
            agg["cut"] += row.get("finish_reason") == "length"
            agg["in"] += row.get("input_tokens", 0)
            agg["out"] += row.get("output_tokens", 0)
            agg["usd"] += row.get("cost_usd", 0.0)
            agg["sec"] += row.get("seconds", 0.0)
    lines = []
    for title, table in (("By model", by_model), ("By tag and model", by_tag)):
        lines.append(title)
        lines.append(f"  {'name':<72}{'calls':>7}{'cut':>5}{'in tok':>12}{'out tok':>10}{'USD':>9}{'sec':>8}")
        for key, agg in sorted(table.items(), key=lambda kv: -kv[1]["usd"]):
            lines.append(
                f"  {key[-71:]:<72}{agg['calls']:>7}{agg['cut'] or '':>5}{agg['in']:>12}{agg['out']:>10}"
                f"{agg['usd']:>9.3f}{agg['sec']:>8.0f}"
            )
        lines.append("")
    total = sum(r.get("cost_usd", 0.0) for r in rows)
    cut = sum(r.get("finish_reason") == "length" for r in rows)
    lines.append(f"Total: {len(rows)} calls, ${total:.3f}" + (f"; {cut} cut off at max_tokens" if cut else ""))
    return "\n".join(lines)


if __name__ == "__main__":  # uv run python -m cs329z_hw1.llm
    print(spending_report())
