"""
One OpenAI client for every harness module, with a single spend ceiling
and usage broken down by purpose (reasoning / perception / reflection /
lesson).

The request shapes mirror openai_agent.OpenAIMarioAgent._call_responses
and _call_chat field for field. That is load-bearing, not tidiness: with
the baseline preset the reasoning request must be byte-identical to the
baseline's, and tools/harness_selftest.py compares the two.
"""

import time
from collections import defaultdict

from openai import OpenAI

import config                     # ../config.py -- OPENAI_* knobs
from harness import settings as S


class BudgetExhausted(RuntimeError):
    """HARNESS_MAX_API_CALLS reached. The run stops cleanly and still writes output."""


class HarnessLLM:

    def __init__(self, client=None):
        self.client = client or OpenAI(timeout=config.OPENAI_TIMEOUT_S, max_retries=0)
        self.usage = defaultdict(lambda: {"calls": 0, "failed": 0, "input_tokens": 0,
                                          "output_tokens": 0, "seconds": 0.0,
                                          "model": None})

    # ------------------------------------------------------------ totals

    @property
    def total_calls(self) -> int:
        return sum(u["calls"] for u in self.usage.values())

    def summary(self) -> dict:
        out = {k: dict(v, seconds=round(v["seconds"], 1)) for k, v in self.usage.items()}
        out["total"] = {
            "calls": self.total_calls,
            "failed": sum(u["failed"] for u in self.usage.values()),
            "input_tokens": sum(u["input_tokens"] for u in self.usage.values()),
            "output_tokens": sum(u["output_tokens"] for u in self.usage.values()),
            "seconds": round(sum(u["seconds"] for u in self.usage.values()), 1),
        }
        return out

    # ---------------------------------------------------------- requests

    def _responses(self, u, model, system, text, images, effort, max_tokens, detail):
        content = [{"type": "input_text", "text": text}]
        for b64 in images:
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}",
                "detail": detail,
            })
        kwargs = dict(
            model=model,
            instructions=system,
            input=[{"role": "user", "content": content}],
            max_output_tokens=max_tokens,
        )
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        resp = self.client.responses.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage:
            u["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
            u["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        return resp.output_text

    def _chat(self, u, model, system, text, images, max_tokens, detail, json_mode):
        if images:
            user_content = [{"type": "text", "text": text}]
            for b64 in images:
                user_content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": detail,
                }})
        else:
            user_content = text
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage:
            u["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            u["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        return resp.choices[0].message.content

    def complete(self, purpose: str, system: str, text: str, images=(),
                 json_mode: bool = True) -> str:
        """One billed call with bounded retries. Raises BudgetExhausted first.

        purpose "reasoning" uses the OPENAI_* settings from ../config.py;
        everything else uses the HARNESS_AUX_* settings.
        """
        if self.total_calls >= S.MAX_API_CALLS:
            raise BudgetExhausted(f"HARNESS_MAX_API_CALLS={S.MAX_API_CALLS} reached")

        if purpose == "reasoning":
            model = config.OPENAI_MODEL
            effort = config.OPENAI_REASONING_EFFORT
            max_tokens = config.OPENAI_MAX_OUTPUT_TOKENS
        else:
            model = S.AUX_MODEL or config.OPENAI_MODEL
            effort = S.AUX_REASONING_EFFORT
            # Same trap as OPENAI_MAX_OUTPUT_TOKENS: reasoning tokens share
            # this budget, and an exhausted budget returns EMPTY, not an error.
            max_tokens = S.AUX_MAX_OUTPUT_TOKENS

        u = self.usage[purpose]
        u["model"] = model
        last_err = None
        for attempt in range(config.OPENAI_MAX_RETRIES):
            t0 = time.time()
            try:
                if config.OPENAI_API_STYLE == "chat":
                    out = self._chat(u, model, system, text, list(images), max_tokens,
                                     config.IMAGE_DETAIL, json_mode)
                else:
                    out = self._responses(u, model, system, text, list(images), effort,
                                          max_tokens, config.IMAGE_DETAIL)
                u["calls"] += 1
                u["seconds"] += time.time() - t0
                return out or ""
            except Exception as exc:          # noqa: BLE001 -- same rationale as openai_agent._call
                u["failed"] += 1
                u["seconds"] += time.time() - t0
                last_err = exc
                backoff = 2 ** attempt
                print(f"  [api:{purpose}] attempt {attempt + 1}/{config.OPENAI_MAX_RETRIES} "
                      f"failed ({type(exc).__name__}: {exc}); retrying in {backoff}s",
                      flush=True)
                time.sleep(backoff)
        raise RuntimeError(f"OpenAI {purpose} call failed after "
                           f"{config.OPENAI_MAX_RETRIES} attempts: {last_err}")
