from typing import List, Dict
from openrouter_client import OpenRouterLLMClient
import os
import json
import re
from collections import Counter


class JudicialPanel:
    """3-Judge deliberative panel for holistic debate evaluation."""

    _JUDGE_SYSTEM_PROMPT = (
        "You are an independent appellate judge presiding over a fact-checking "
        "legal proceeding. Your role is to perform a comprehensive holistic "
        "evaluation of the case, focusing on evidence admissibility, logical "
        "coherence of advocacy, and reliability of the sources and expert "
        "testimonies — regardless of whether the claim concerns medicine, "
        "sports, history, politics, or any other domain.\n\n"
        "CRITICAL VERDICT RULE — INCONCLUSIVE USAGE:\n"
        "INCONCLUSIVE is only permitted when evidence is genuinely absent from "
        "the record OR when the evidence is fundamentally contradictory with no "
        "clear preponderance for either side.\n"
        "If you can identify which side has presented the stronger case from "
        "the evidence before you, you MUST return SUPPORTED or NOT SUPPORTED "
        "even if you are not perfectly certain. Reserve INCONCLUSIVE for cases "
        "where it is truly impossible to determine which argument is stronger.\n\n"
        "WIKIPEDIA SOURCE RULE:\n"
        "For encyclopaedic fact-checking proceedings (sports records, biographical "
        "facts, taxonomic classification, historical events, geographical facts), "
        "Wikipedia articles ARE the primary authoritative source. Do NOT penalise "
        "evidence solely because it originates from Wikipedia. Evaluate whether the "
        "Wikipedia article is specific, named, and provides relevant factual content."
    )

    def __init__(self):
        self.judges = [
            {
                "name":  "Judge 1",
                "llm":   OpenRouterLLMClient(
                    model_name="deepseek/deepseek-r1",
                    system_prompt=self._JUDGE_SYSTEM_PROMPT,
                    temperature=0.3),
                "model": "deepseek/deepseek-r1",
            },
            {
                "name":  "Judge 2",
                "llm":   OpenRouterLLMClient(
                    model_name="nousresearch/hermes-3-llama-3.1-405b",
                    system_prompt=self._JUDGE_SYSTEM_PROMPT,
                    temperature=0.3),
                "model": "nousresearch/hermes-3-llama-3.1-405b",
            },
            {
                "name":  "Judge 3",
                "llm":   OpenRouterLLMClient(
                    model_name="qwen/qwen3-235b-a22b-2507",
                    system_prompt=self._JUDGE_SYSTEM_PROMPT,
                    temperature=0.3),
                "model": "qwen/qwen3-235b-a22b-2507",
            },
        ]

    # ─────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────

    def evaluate_debate(self, debate_transcript: Dict,
                        admitted_evidence: List = None,
                        role_switch_history: Dict = None,
                        prag_metrics: Dict = None,
                        critic_evaluations: List[Dict] = None,
                        reflection_history: List[Dict] = None) -> Dict:

        print("\n" + "="*60)
        print("JUDICIAL PANEL EVALUATION")
        print("="*60 + "\n")

        claim            = debate_transcript['claim']
        proponent_args   = self._extract_side_arguments(debate_transcript, 'proponent')
        opponent_args    = self._extract_side_arguments(debate_transcript, 'opponent')
        evidence_summary = self._extract_evidence_summary(
            debate_transcript, admitted_evidence)
        role_switch_summary = (self._format_role_switch(role_switch_history)
                                if role_switch_history
                                else "No role-switching performed.")

        judge_verdicts = []
        for judge in self.judges:
            print(f"{judge['name']} ({judge['model']}) deliberating...")
            verdict = self._judge_evaluate(
                judge, claim, proponent_args, opponent_args,
                evidence_summary, role_switch_summary, debate_transcript,
                prag_metrics, critic_evaluations, reflection_history)
            judge_verdicts.append(verdict)
            print(f"  Verdict: {verdict['verdict']}")
            print(f"  Evidence Strength:   {verdict['evidence_strength']}/10")
            print(f"  Argument Validity:   {verdict['argument_validity']}/10")
            print(f"  Source Reliability:  {verdict['source_reliability']}/10\n")

        aggregation = self._aggregate_verdicts(judge_verdicts)

        result = {
            "claim":              claim,
            "judge_verdicts":     judge_verdicts,
            "final_verdict":      aggregation['final_verdict'],
            "majority_opinion":   aggregation['majority_opinion'],
            "dissenting_opinion": aggregation['dissenting_opinion'],
            "vote_breakdown":     aggregation['vote_breakdown'],
        }

        try:
            from logging_extension import append_framework_json
            append_framework_json("judge_evaluation.jsonl", claim, result)
        except ImportError:
            with open("judge_evaluation.json", "w") as f:
                json.dump(result, f, indent=2)

        print(f"\nFinal Verdict: {aggregation['final_verdict']}")
        print(f"Vote Breakdown: {aggregation['vote_breakdown']}")
        if aggregation['dissenting_opinion']:
            print("Dissent Present: Yes")

        return result

    # ─────────────────────────────────────────────────────────────────
    # Single-judge evaluation
    # ─────────────────────────────────────────────────────────────────

    def _judge_evaluate(self, judge: Dict, claim: str,
                        proponent_args: List[str], opponent_args: List[str],
                        evidence_summary: str, role_switch_summary: str,
                        full_transcript: Dict, prag_metrics: Dict = None,
                        critic_evaluations: List[Dict] = None,
                        reflection_history: List[Dict] = None) -> Dict:

        prompt = f"""You are an appellate judge evaluating the following fact-checking proceedings.

PROCEEDINGS RECORD
══════════════════
CLAIM: {claim}

PLAINTIFF COUNSEL'S ARGUMENTS:
{chr(10).join(proponent_args)}

DEFENSE COUNSEL'S ARGUMENTS:
{chr(10).join(opponent_args)}

ADMITTED EVIDENCE & EXPERT TESTIMONIES:
{evidence_summary}

ROLE-SWITCH HISTORY (ADVERSARY CONSISTENCY):
{role_switch_summary}

EVIDENCE DISCOVERY METRICS (PRAG EVOLUTION):
{json.dumps(prag_metrics, indent=2) if prag_metrics else "No P-RAG data available."}

INDEPENDENT CRITIC EVALUATIONS:
{json.dumps(critic_evaluations, indent=2) if critic_evaluations else "No critic data."}

AGENT SELF-REFLECTION TRENDS:
{json.dumps(reflection_history, indent=2) if reflection_history else "No reflection data."}

══════════════════════════════════════════════════════════════
EVALUATION STAGES
══════════════════════════════════════════════════════════════

STAGE 1 – CASE RECONSTRUCTION
Identify:
- Core factual claim being adjudicated
- Main supporting arguments (Plaintiff)
- Main counterarguments (Defense)

STAGE 2 – EVIDENCE & TESTIMONY WEIGHTING
Score: Evidence Strength (0–10)
  7–10: Strong, specific, directly verifiable evidence
  4–6 : Moderate evidence with some gaps or ambiguity
  0–3 : Weak, irrelevant, or missing evidence

STAGE 3 – LOGICAL COHERENCE
Score: Argument Validity (0–10)
  7–10: Sound reasoning, minimal logical flaws
  4–6 : Generally coherent with some logical issues
  0–3 : Multiple fallacies or unsupported inferences

STAGE 4 – SOURCE RELIABILITY
(Applies to ALL domains, not just medical/scientific claims.)

Score: Source Reliability (0–10)
  7–10: Named, verifiable sources with specific facts (official records,
        contemporary news, university archives, peer-reviewed papers,
        government records, specific named Wikipedia articles with facts)
  4–6 : General reference sources, secondary summaries, or sources lacking
        specific details
  0–3 : Unsourced assertions, anonymous content, or clearly speculative claims

IMPORTANT SOURCE NOTE: For encyclopaedic fact-checking (sports records,
biographical facts, taxonomy, historical events), Wikipedia IS the primary
authoritative source. Do NOT score Wikipedia evidence as low-reliability
solely because it is Wikipedia. Score based on whether the cited Wikipedia
article is specific, named, and provides relevant factual content.

STAGE 5 – DISCOVERY RIGOR & TRANSPARENCY
Analyse P-RAG metrics:
- Query evolution and evidence novelty over rounds
- Court query refinement impact on evidence quality

STAGE 6 – JUDICIAL VERDICT

MANDATORY DECISION RULE:
  SUPPORTED     : Claim is supported by the available evidence and arguments
  NOT SUPPORTED : Claim is not adequately supported or is refuted
  INCONCLUSIVE  : Use ONLY if evidence is genuinely absent OR if evidence is
                  fundamentally contradictory with no clear preponderance.
                  If one side has clearly presented stronger evidence or
                  arguments, you MUST choose SUPPORTED or NOT SUPPORTED.
                  Do NOT use INCONCLUSIVE as a hedge when the evidence
                  points in one direction, even imperfectly.

Respond ONLY in valid JSON — no markdown, no preamble:
{{
  "claim_summary": "Brief summary of the core claim and debate",
  "evidence_strength": <integer 0-10>,
  "argument_validity": <integer 0-10>,
  "source_reliability": <integer 0-10>,
  "verdict": "SUPPORTED" or "NOT SUPPORTED" or "INCONCLUSIVE",
  "reasoning": "2-3 sentence justification citing specific evidence and which side had the stronger case"
}}"""

        response = judge['llm'].generate(prompt)

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                raise ValueError("No JSON in response")
            vd = json.loads(json_match.group())

            # Backward compat: old key name
            if "scientific_reliability" in vd and "source_reliability" not in vd:
                vd["source_reliability"] = vd.pop("scientific_reliability")

            required = ['claim_summary', 'evidence_strength', 'argument_validity',
                        'source_reliability', 'verdict', 'reasoning']
            for f in required:
                if f not in vd:
                    raise ValueError(f"Missing field: {f}")

            if vd['verdict'] not in ['SUPPORTED', 'NOT SUPPORTED', 'INCONCLUSIVE']:
                vd['verdict'] = 'INCONCLUSIVE'

            for sf in ['evidence_strength', 'argument_validity', 'source_reliability']:
                vd[sf] = max(0, min(10, int(vd[sf])))

        except Exception as e:
            print(f"  [WARNING] Failed to parse judge response: {e}")
            vd = {
                "claim_summary":     f"Evaluation of: {claim}",
                "evidence_strength": 5,
                "argument_validity": 5,
                "source_reliability":5,
                "verdict":           "INCONCLUSIVE",
                "reasoning":         "Unable to parse structured evaluation.",
            }

        vd['judge_name'] = judge['name']
        vd['model']      = judge['model']
        return vd

    # ─────────────────────────────────────────────────────────────────
    # Aggregation
    # ─────────────────────────────────────────────────────────────────

    def _aggregate_verdicts(self, judge_verdicts: List[Dict]) -> Dict:
        vote_counts    = Counter(v['verdict'] for v in judge_verdicts)
        final_verdict  = vote_counts.most_common(1)[0][0]
        majority_judges  = [v for v in judge_verdicts if v['verdict'] == final_verdict]
        dissenting_judges= [v for v in judge_verdicts if v['verdict'] != final_verdict]

        return {
            "final_verdict":      final_verdict,
            "majority_opinion":   self._synthesize_opinion(majority_judges, "majority"),
            "dissenting_opinion": (self._synthesize_opinion(dissenting_judges, "dissent")
                                   if dissenting_judges else None),
            "vote_breakdown":     dict(vote_counts),
        }

    def _synthesize_opinion(self, judges: List[Dict], opinion_type: str) -> str:
        if not judges:
            return ""
        if len(judges) == 1:
            j = judges[0]
            return f"{j['judge_name']} ({j['model']}) — {j['verdict']}: {j['reasoning']}"
        verdict     = judges[0]['verdict']
        judge_names = ", ".join(j['judge_name'] for j in judges)
        lines       = [f"- {j['judge_name']}: {j['reasoning']}" for j in judges]
        return (f"{opinion_type.capitalize()} Opinion ({judge_names}) — {verdict}:\n\n"
                + "\n".join(lines))

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _extract_side_arguments(self, transcript: Dict, role: str) -> List[str]:
        args = []
        for rd in transcript['rounds']:
            for arg in rd['arguments']:
                if arg['role'] == role:
                    args.append(arg['text'])
            for exp in rd.get('expert_testimonies', []):
                if exp.get('requesting_side') == role:
                    args.append(f"[Expert Testimony]: {exp['text']}")
        return args

    def _extract_evidence_summary(self, transcript: Dict,
                                   admitted_evidence: List) -> str:
        if not admitted_evidence:
            ids = set()
            for rd in transcript.get('rounds', []):
                for arg in rd.get('arguments', []):
                    ids.update(re.findall(r'\b\d{7,9}\b', arg['text']))
            return (f"Evidence sources cited: {', '.join(sorted(ids))}"
                    if ids else "No specific evidence sources identified.")

        lines = []
        for i, ev in enumerate(admitted_evidence[:10], 1):
            sid  = (ev.source_id if hasattr(ev, 'source_id')
                    else ev.get('source_id', 'unknown'))
            text = (ev.text[:150] if hasattr(ev, 'text')
                    else ev.get('text', '')[:150])
            lines.append(f"{i}. Source {sid}: {text}...")
        return "\n".join(lines)

    def _format_role_switch(self, rsh: Dict) -> str:
        if not rsh:
            return "No role-switching performed."
        return f"Role-Switching Consistency Analysis:\n{rsh.get('analysis', 'N/A')}"