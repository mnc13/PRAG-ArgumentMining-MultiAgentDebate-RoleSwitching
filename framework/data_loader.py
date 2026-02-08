import json
import os
from typing import List, Dict
from models import Claim

class DataLoader:
    def __init__(self, check_covid_dir: str):
        self.check_covid_dir = check_covid_dir
        self.claims_path = os.path.join(check_covid_dir, "Check-COVID_all.jsonl")
        self.corpus_path = os.path.join(check_covid_dir, "corpus.json")

    def load_claims(self, limit: int = 5) -> List[Claim]:
        """Loads a limited number of claims for testing."""
        claims = []
        try:
            with open(self.claims_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(claims) >= limit:
                        break
                    data = json.loads(line)
                    claims.append(Claim(
                        id=data['id'],
                        text=data['claim'],
                        metadata={'cord_id': data['cord_id'], 'label': data['label']}
                    ))
        except FileNotFoundError:
            print(f"Error: Claims file not found at {self.claims_path}")
        return claims

    def load_corpus(self) -> Dict[str, Dict]:
        """Loads the corpus into a dictionary keyed by cord_id."""
        corpus = {}
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    corpus[data['cord_id']] = {
                        'title': data['title'],
                        'abstract': data['abstract']
                    }
        except FileNotFoundError:
            print(f"Error: Corpus file not found at {self.corpus_path}")
        return corpus
