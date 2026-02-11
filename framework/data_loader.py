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

    def load_specific_file(self, file_path: str) -> List[Claim]:
        """Loads claims from a specific JSON file."""
        claims = []
        try:
            # Handle absolute or relative paths
            if os.path.isabs(file_path):
                full_path = file_path
            else:
                full_path = os.path.join(self.check_covid_dir, file_path)
                
            print(f"Loading from: {full_path}")
            
            with open(full_path, 'r', encoding='utf-8') as f:
                # Check if it's a JSON list or JSONL
                try:
                    content = f.read().strip()
                    if not content:
                         return []
                    f.seek(0)
                    
                    # Try standard JSON first
                    data = json.load(f)
                    if isinstance(data, list):
                        # It is a list of claims
                        for entry in data:
                             claims.append(Claim(
                                id=str(entry.get('id', 'unknown')), 
                                text=entry.get('claim', ''),
                                metadata={'label': entry.get('label', ''), 'evidence': entry.get('evidence', [])}
                            ))
                    elif isinstance(data, dict):
                         # Handle single object
                         claims.append(Claim(
                                id=str(data.get('id', 'unknown')), 
                                text=data.get('claim', ''),
                                metadata={'label': data.get('label', ''), 'evidence': data.get('evidence', [])}
                        ))
                    else:
                        print("Error: JSON file content is not a list or dict.")
                except json.JSONDecodeError:
                    # Try JSONL if standard load fails
                    f.seek(0)
                    for line in f:
                        if not line.strip(): continue
                        entry = json.loads(line)
                        claims.append(Claim(
                            id=str(entry.get('id', 'unknown')),
                            text=entry.get('claim', ''),
                            metadata={'label': entry.get('label', '')}
                        ))

        except FileNotFoundError:
             print(f"Error: Specific file not found at {file_path}")
        except Exception as e:
            print(f"Error loading specific file: {e}")
            
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
