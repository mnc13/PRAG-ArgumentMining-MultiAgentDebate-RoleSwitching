import sys
import os
import re
import time
from openai import OpenAI


class DualLogger:
    def __init__(self, log_dir, claim_id):
        self.log_file_path = os.path.join(log_dir, f"log_gpt_{claim_id}.txt")
        self.terminal = sys.stdout
        self.log_file = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.terminal
        if self.log_file:
            self.log_file.close()

    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()


_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")

def _is_reasoning_model(model_name: str) -> bool:
    name = model_name.lower()
    return any(name.startswith(p) or f"/{p}" in name for p in _REASONING_MODEL_PREFIXES)


class GPTClient:
    """
    Wrapper for OpenAI chat models.

    gpt-5-mini is a reasoning model: it burns tokens on internal chain-of-thought
    BEFORE writing visible output. max_completion_tokens covers BOTH reasoning +
    output tokens combined. If the budget is too small, reasoning fills it up and
    content is empty. We therefore use large budgets:
      - miner  : 8 000  tokens  (reasoning ~6 k, list output ~500)
      - verdict: 16 000 tokens  (reasoning ~12 k, analysis ~2 k)
    These defaults are set by the callers; GPTClient just forwards whatever it receives.
    """

    def __init__(self, model_name="gpt-5-mini", temperature=0.1, max_tokens=16000):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: OPENAI_API_KEY not found in environment variables", file=sys.stderr)
        self.client      = OpenAI(api_key=api_key)
        self.model_name  = model_name
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._reasoning  = _is_reasoning_model(model_name)
        print(
            f"[GPTClient] model='{model_name}'  reasoning_mode={self._reasoning}"
            f"  max_tokens={max_tokens}",
            file=sys.stderr,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        max_retries = 5
        base_delay  = 2

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        call_kwargs: dict = {"model": self.model_name, "messages": messages}
        if self._reasoning:
            # Reasoning models: no temperature, use max_completion_tokens
            call_kwargs["max_completion_tokens"] = self.max_tokens
        else:
            call_kwargs["temperature"] = self.temperature
            call_kwargs["max_tokens"]  = self.max_tokens

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**call_kwargs)

                finish = response.choices[0].finish_reason
                usage  = response.usage
                tokens = usage.total_tokens if usage else 0

                # Log token breakdown so we can spot budget exhaustion immediately
                if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                    det = usage.completion_tokens_details
                    r_tok = getattr(det, "reasoning_tokens", "?")
                    c_tok = getattr(usage, "completion_tokens", "?")
                    print(
                        f"[GPTClient] finish={finish!r}  total={tokens}"
                        f"  reasoning={r_tok}  completion={c_tok}",
                        file=sys.stderr,
                    )
                else:
                    print(f"[GPTClient] finish={finish!r}  total={tokens}", file=sys.stderr)

                content = response.choices[0].message.content or ""

                # Warn if budget was exhausted — caller should raise max_tokens
                if finish == "length" and not content.strip():
                    print(
                        f"[GPTClient] WARNING: finish_reason='length' and content empty. "
                        f"Reasoning consumed the entire {self.max_tokens}-token budget. "
                        f"Increase max_tokens.",
                        file=sys.stderr,
                    )

                return {"content": content.strip(), "tokens": tokens, "error": None}

            except Exception as e:
                err_msg = str(e)
                print(
                    f"[GPTClient] Error ({self.model_name}) "
                    f"- Attempt {attempt + 1}/{max_retries}: {err_msg}",
                    file=sys.stderr,
                )
                # Auto-correct param mismatches
                if "max_completion_tokens" in err_msg and "max_tokens" in call_kwargs:
                    call_kwargs.pop("max_tokens", None)
                    call_kwargs.pop("temperature", None)
                    call_kwargs["max_completion_tokens"] = self.max_tokens
                    self._reasoning = True
                    continue
                if "max_tokens" in err_msg and "max_completion_tokens" in call_kwargs:
                    call_kwargs.pop("max_completion_tokens", None)
                    call_kwargs["max_tokens"]  = self.max_tokens
                    call_kwargs["temperature"] = self.temperature
                    self._reasoning = False
                    continue
                if "temperature" in err_msg:
                    call_kwargs.pop("temperature", None)
                    self._reasoning = True
                    continue

                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    return {"content": "", "tokens": 0, "error": err_msg}

        return {"content": "", "tokens": 0, "error": "Max retries exceeded"}


# ---------------------------------------------------------------------------
# Plain-text verdict parser
# ---------------------------------------------------------------------------

def parse_verdict_response(text: str) -> dict:
    verdict    = "UNKNOWN"
    confidence = 0.0
    reasoning  = ""

    m = re.search(r'FINAL VERDICT\s*:\s*(SUPPORT|REFUTE)', text, re.IGNORECASE)
    if m:
        verdict = m.group(1).upper()

    m = re.search(r'CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
    if m:
        try:
            confidence = float(m.group(1))
            if confidence > 1.0:
                confidence = confidence / 100.0
        except ValueError:
            confidence = 0.0

    m = re.search(r'REASONING\s*:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    if m:
        reasoning = m.group(1).strip()

    return {"verdict": verdict, "confidence": confidence, "reasoning": reasoning}