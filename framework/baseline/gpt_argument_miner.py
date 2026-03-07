import re
from gpt_utils import GPTClient


class GPTArgumentMiner:
    def __init__(self, model_name="gpt-5-mini"):
        # 8000 tokens: gives reasoning model ~6-7k for chain-of-thought,
        # leaving plenty of budget for a short numbered list as output.
        self.client = GPTClient(model_name=model_name, temperature=0.1, max_tokens=8000)

        self.system_prompt = (
            "You are a biomedical research assistant. "
            "Decompose the given claim into atomic, testable sub-premises "
            "suitable for evidence retrieval from PubMed. "
            "Output ONLY a numbered list — no introduction, no explanation. "
            "Each line must start with a number, a period, and a space. Example:\n"
            "1. First premise.\n"
            "2. Second premise."
        )

    def mine_arguments(self, claim_text: str):
        user_prompt = (
            f'Decompose this clinical claim into atomic, testable premises:\n"{claim_text}"\n\n'
            "Rules:\n"
            "- Output ONLY a numbered list, nothing else.\n"
            "- Each line: number, period, space, one sentence.\n"
            "- Generate exactly 6 to 8 premises.\n"
            "- No introduction, no summary, no explanation."
        )

        result  = self.client.generate(self.system_prompt, user_prompt)
        content = result.get("content", "") or ""
        tokens  = result.get("tokens", 0)

        if result.get("error"):
            print(f"  [GPTArgumentMiner] API error: {result['error']}")

        premises = []
        for line in content.split("\n"):
            line = line.strip()
            if re.match(r'^(\d+[\.\)]|-|\*|•)\s+', line):
                premise = re.sub(r'^(\d+[\.\)]|-|\*|•)\s+', '', line).strip()
                if premise:
                    premises.append(premise)

        if not premises and content.strip():
            raw      = re.split(r'(?<=[.!?])\s+', content)
            premises = [s.strip() for s in raw if len(s.strip()) > 20][:8]

        if not premises:
            print("  [GPTArgumentMiner] Could not extract premises; falling back to claim text.")
            premises = [claim_text]

        print(f"  [GPTArgumentMiner] Extracted {len(premises)} premises ({len(content)} chars)")
        return premises, tokens