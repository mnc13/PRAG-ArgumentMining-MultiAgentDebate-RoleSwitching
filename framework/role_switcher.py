"""
Role-Switching Mechanism for Consistency Testing  v2

Key fix:
  - The consistency-analyzer LLM was created via create_llm_client() which
    uses base AGENT_SLOTS (no domain notice), causing it to reason in a
    generic/medical legal framing.  Now uses get_agent_slots(claim_text) to
    inject the correct domain notice so the analyzer stays on-topic.

  - The swapped DebateAgent objects already carry domain-enriched system
    prompts from MADOrchestrator.__init__() → get_agent_slots(claim.text),
    so the actual debate agents are fine after swapping.  Only the external
    consistency-analyzer needed fixing.
"""

import json
from typing import Dict
from mad_orchestrator import MADOrchestrator


class RoleSwitcher:
    """Manages role-switching and consistency analysis."""

    def __init__(self, mad_orchestrator: MADOrchestrator):
        self.original_mad = mad_orchestrator

    # ─────────────────────────────────────────────────────────────────
    # Switch & re-run
    # ─────────────────────────────────────────────────────────────────

    def switch_roles(self, max_rounds: int = 3) -> Dict:
        """
        Swap Plaintiff Counsel ↔ Defense Counsel roles and re-run debate.

        The swapped agents already carry domain-enriched system prompts
        (set when MADOrchestrator was first constructed with the claim text),
        so no extra work is needed here — just swap, reset, and run.
        """
        print("\n" + "="*60)
        print("ROLE-SWITCHING ROUND")
        print("="*60)
        print("Swapping Plaintiff Counsel ↔ Defense Counsel roles...")
        print()

        original_proponent = self.original_mad.agents['proponent']
        original_opponent  = self.original_mad.agents['opponent']

        # Swap
        self.original_mad.agents['proponent'] = original_opponent
        self.original_mad.agents['opponent']  = original_proponent

        # Update role metadata on the swapped agents
        self.original_mad.agents['proponent'].role      = 'proponent'
        self.original_mad.agents['proponent'].job_title = "Plaintiff Counsel"
        self.original_mad.agents['proponent'].name      = "Plaintiff Counsel"

        self.original_mad.agents['opponent'].role      = 'opponent'
        self.original_mad.agents['opponent'].job_title = "Defense Counsel"
        self.original_mad.agents['opponent'].name      = "Defense Counsel"

        # Reset debate state (keeps PRAG pool seeded with negotiation evidence)
        self.original_mad.reset_state()

        # Run switched debate
        switched_result = self.original_mad.run_full_debate(
            max_rounds=max_rounds,
            save_transcript=True,
            file_suffix="_switched",
        )

        return switched_result

    # ─────────────────────────────────────────────────────────────────
    # Consistency check
    # ─────────────────────────────────────────────────────────────────

    def check_consistency(self, original_transcript: Dict,
                          switched_transcript: Dict) -> Dict:
        """
        Analyse logical consistency between original and switched debates.

        FIX: analyzer now uses get_agent_slots(claim_text) so its system
        prompt includes the domain notice and doesn't drift into medical
        framing.
        """
        print("\n" + "="*60)
        print("CONSISTENCY ANALYSIS")
        print("="*60 + "\n")

        # ── Build domain-aware analyzer ───────────────────────────────
        claim_text = original_transcript.get('claim', '')

        from personas import get_agent_slots, create_llm_client
        slots = get_agent_slots(claim_text)

        analyzer = create_llm_client({
            "llm_provider": "openrouter",
            "llm_model":    "deepseek/deepseek-chat",
            "temperature":  0.3,
            "system_prompt": (
                "You are an expert in logical consistency analysis and "
                "argumentation theory. " + slots['critic']['system_prompt']
            ),
            "name":      "Consistency Analyzer",
            "role":      "Consistency Analyzer",
            "expertise": ["logic", "argumentation"],
        })

        # ── Extract arguments ─────────────────────────────────────────
        def _args(transcript, role):
            return [
                arg['text']
                for rd in transcript.get('rounds', [])
                for arg in rd.get('arguments', [])
                if arg['role'] == role
            ]

        original_pro_args  = _args(original_transcript,  'proponent')
        original_opp_args  = _args(original_transcript,  'opponent')
        switched_pro_args  = _args(switched_transcript,  'proponent')
        switched_opp_args  = _args(switched_transcript,  'opponent')

        # ── Prompt ───────────────────────────────────────────────────
        prompt = f"""Analyse the logical consistency of arguments when agents switch roles.

CLAIM BEING ADJUDICATED:
{claim_text}

ORIGINAL PROCEEDINGS:
Plaintiff Counsel (Agent A) Arguments:
{chr(10).join(original_pro_args)}

Defense Counsel (Agent B) Arguments:
{chr(10).join(original_opp_args)}

SWITCHED PROCEEDINGS (Roles Swapped):
Plaintiff Counsel (Agent B - formerly Defense) Arguments:
{chr(10).join(switched_pro_args)}

Defense Counsel (Agent A - formerly Plaintiff) Arguments:
{chr(10).join(switched_opp_args)}

Analyse:
1. Does Agent A maintain logical consistency when switching from Plaintiff Counsel to Defense Counsel?
2. Does Agent B maintain logical consistency when switching from Defense Counsel to Plaintiff Counsel?
3. Are there contradictions in their arguments?
4. Overall consistency score (0-10)

Respond ONLY in valid JSON — no markdown, no preamble:
{{
  "agent_a_analysis": "...",
  "agent_b_analysis": "...",
  "contradictions_found": "...",
  "consistency_score": <integer 0-10>,
  "is_consistent": <true if consistency_score >= 6, else false>,
  "reasoning": "..."
}}"""

        analysis = analyzer.generate(prompt)

        # ── Parse ────────────────────────────────────────────────────
        raw_json = analysis.strip()
        for prefix in ("```json", "```"):
            if raw_json.startswith(prefix):
                raw_json = raw_json[len(prefix):]
        if raw_json.endswith("```"):
            raw_json = raw_json[:-3]

        try:
            parsed            = json.loads(raw_json)
            consistency_score = parsed.get("consistency_score", 5)
            is_consistent     = parsed.get("is_consistent", False)
            reasoning         = parsed.get("reasoning", str(analysis))
            analysis_data     = parsed
        except Exception as e:
            print(f"Warning: Failed to parse consistency JSON: {e}")
            consistency_score = 5
            is_consistent     = False
            reasoning         = str(analysis)
            analysis_data     = analysis

        consistency_report = {
            "claim":            original_transcript.get('claim', 'unknown'),
            "claim_id":         original_transcript.get('claim_id', 'unknown'),
            "original_agents":  original_transcript.get('agents', {}),
            "switched_agents":  switched_transcript.get('agents', {}),
            "analysis":         analysis_data,
            "consistency_score":consistency_score,
            "is_consistent":    is_consistent,
            "reasoning":        reasoning,
            "original_rounds":  len(original_transcript.get('rounds', [])),
            "switched_rounds":  len(switched_transcript.get('rounds', [])),
        }

        # ── Persist ──────────────────────────────────────────────────
        try:
            from logging_extension import append_framework_json
            append_framework_json(
                "role_switch_report.jsonl",
                self.original_mad.claim,
                consistency_report,
            )
        except ImportError:
            with open("role_switch_report.json", "w") as f:
                json.dump(consistency_report, f, indent=2)

        print(f"Consistency Analysis:\n{analysis}\n")
        return consistency_report