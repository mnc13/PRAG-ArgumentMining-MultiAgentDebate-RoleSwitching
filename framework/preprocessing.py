from models import Claim

class ClaimExtractor:
    def extract_claim(self, input_data: str) -> Claim:
        """
        Simulates extracting a claim from an input text.
        For now, it assumes the input IS the claim.
        """
        # In a real scenario, this might use an LLM or NLP to parse an article.
        return Claim(id="extracted_001", text=input_data)
