import sys
import os
import json
import requests
import time
import re

class DualLogger:
    def __init__(self, log_dir, claim_id):
        self.log_file_path = os.path.join(log_dir, f"log_{claim_id}.txt")
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


class BaselineOpenRouterClient:
    def __init__(self, model_name="deepseek/deepseek-v3.2", temperature=0.1, max_tokens=512):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            print("WARNING: OPENROUTER_API_KEY not found in environment variables", file=sys.stderr)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, system_prompt, user_prompt, request_logprobs=False, enable_reasoning=False):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        if "deepseek" in self.model_name.lower() and enable_reasoning:
            data["reasoning"] = {"enabled": True}

        if request_logprobs:
            data["top_logprobs"] = 5
            data["logprobs"] = True

        max_retries = 5
        base_delay = 2

        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=data, timeout=120)
                response.raise_for_status()
                result = response.json()

                message = result['choices'][0]['message']

                # Primary content field
                content = message.get('content', '')
                if content is None:
                    content = ''

                # --- KEY FIX: DeepSeek reasoning models can return an empty `content`
                # and put the actual answer in `reasoning_content`. Fall back to it.
                if not content.strip():
                    reasoning_content = message.get('reasoning_content', '')
                    if reasoning_content and reasoning_content.strip():
                        print(
                            f"[BaselineOpenRouterClient] 'content' was empty; "
                            f"falling back to 'reasoning_content' ({len(reasoning_content)} chars)."
                        )
                        content = reasoning_content

                # Strip any <think> wrapper that may still be present
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

                usage = result.get('usage', {}).get('total_tokens', 0)

                logprobs_data = None
                if (
                    request_logprobs
                    and 'logprobs' in result['choices'][0]
                    and result['choices'][0]['logprobs']
                ):
                    logprobs_data = result['choices'][0]['logprobs'].get('content', [])

                return {
                    "content": content,
                    "tokens": usage,
                    "logprobs": logprobs_data
                }

            except Exception as e:
                err_txt = ""
                if hasattr(e, 'response') and e.response is not None:
                    err_txt = f" - Body: {e.response.text}"
                print(
                    f"Error calling OpenRouter ({self.model_name}) "
                    f"- Attempt {attempt + 1}/{max_retries}: {e}{err_txt}"
                )
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    return {
                        "content": "",
                        "tokens": 0,
                        "logprobs": None,
                        "error": str(e)
                    }


def safe_parse_json(content, client, system_prompt, user_prompt):
    def extract_json(text):
        if text is None:
            text = ""
        text = text.strip()
        # Strip <think> blocks just in case
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if "```json" in text:
            parts = text.split("```json")
            if len(parts) > 1:
                text = parts[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                text = parts[1].split("```")[0].strip()
        # Find first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end + 1]
        if not text.strip():
            raise ValueError("Empty JSON response")
        return json.loads(text)

    try:
        return extract_json(content)
    except Exception as e:
        print(f"JSON parse error: {e}. Retrying with explicit instruction...")
        retry_prompt = (
            user_prompt
            + "\n\nCRITICAL INSTRUCTION: Respond ONLY with valid JSON, "
            "no markdown formatting. Do not include any explanations or backticks. "
            "Do NOT emit <think> reasoning — output the JSON object directly."
        )
        result = client.generate(system_prompt, retry_prompt, request_logprobs=True, enable_reasoning=False)
        try:
            if result.get("error"):
                raise Exception(result["error"])
            parsed = extract_json(result["content"])
            return parsed, result
        except Exception as e2:
            print(f"Failed again: {e2}")
            return None, result