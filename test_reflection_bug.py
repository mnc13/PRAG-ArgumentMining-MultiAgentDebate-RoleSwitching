
from framework.self_reflection import SelfReflection
import json

class MockLLM:
    def generate(self, prompt):
        return '{"scores": {"logic": 0.9, "novelty": 0.8, "rebuttal": 0.7}, "discovery_need": "test"}'

class MockAgent:
    def __init__(self):
        self.llm = MockLLM()
        self.name = "Mock Agent"
        self.job_title = "Mock Counselor"

def test():
    transcript = []
    sr = SelfReflection(transcript)
    agent = MockAgent()
    res = sr.perform_round_reflection(agent, "proponent", 1, "Testing claim")
    print(f"Result: {res}")
    sr.save_reflection_history("test_reflection.json")
    print("Saved history.")

if __name__ == "__main__":
    test()
