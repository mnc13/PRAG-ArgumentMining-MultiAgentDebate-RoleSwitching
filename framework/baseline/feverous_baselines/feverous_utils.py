import sys
import os
import json

# Ensure parent dirs are in path so we can import from framework
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import Claim

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

def load_feverous_data(file_path):
    claims = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            claim = Claim(
                id=str(data.get("id")),
                text=data.get("claim"),
                metadata={
                    "label": data.get("label"),
                    "cord_id": data.get("cord_id")
                }
            )
            claims.append(claim)
    return claims
