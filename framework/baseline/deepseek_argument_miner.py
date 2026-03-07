from deepseek_utils import BaselineOpenRouterClient
import re

class BaselineArgumentMiner:
    def __init__(self, model_name="deepseek/deepseek-v3.2"):
        self.client = BaselineOpenRouterClient(model_name=model_name, temperature=0.1, max_tokens=512)
        self.system_prompt = (
            "You are a biomedical research assistant. "
            "Decompose the given claim into atomic, testable sub-premises suitable for evidence retrieval from PubMed. "
            "IMPORTANT: Output ONLY a numbered list. No preamble, no explanation, no markdown headers. "
            "Each line must start with a number followed by a period and a space, e.g.:\n"
            "1. First premise here.\n"
            "2. Second premise here."
        )

    def mine_arguments(self, claim_text):
        user_prompt = (
            f'Decompose this clinical claim into atomic, testable premises:\n"{claim_text}"\n\n'
            "Rules:\n"
            "- Output ONLY a numbered list, nothing else.\n"
            "- Each line: a number, a period, a space, then one sentence.\n"
            "- Generate exactly 6 to 8 premises.\n"
            "- Do NOT include any introduction, summary, or explanation.\n\n"
            "Example format:\n"
            "1. Premise one.\n"
            "2. Premise two.\n"
            "3. Premise three."
        )

        result = self.client.generate(self.system_prompt, user_prompt, request_logprobs=False)
        content = result.get('content', '') or ''
        tokens = result.get('tokens', 0)

        # Strip <think>...</think> blocks that DeepSeek reasoning models emit
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        premises = []
        for line in content.split('\n'):
            line = line.strip()
            # Match lines starting with: 1. / 1) / - / * / •
            if re.match(r'^(\d+[\.\)]|-|\*|•)\s+', line):
                # Strip ALL of those prefixes uniformly
                premise = re.sub(r'^(\d+[\.\)]|-|\*|•)\s+', '', line).strip()
                if premise:
                    premises.append(premise)

        # Fallback: split by sentence boundaries if list parsing failed
        if not premises and content.strip():
            raw_sentences = re.split(r'(?<=[.!?])\s+', content)
            premises = [s.strip() for s in raw_sentences if len(s.strip()) > 20][:8]

        # Last-resort fallback
        if not premises:
            premises = [claim_text]

        print(f"  [ArgumentMiner] Extracted {len(premises)} premises from content ({len(content)} chars)")
        return premises, tokens