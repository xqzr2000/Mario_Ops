"""
Reasoning module: assemble everything into one request, get a plan back.

Reuses the baseline wherever the baseline is already right:
  * telemetry   OpenAIMarioAgent.build_telemetry, unchanged
  * parsing     OpenAIMarioAgent.parse_plan -- clamping, fence tolerance,
                the MAX_*_PER_PLAN limits -- unchanged
  * fallback    the same [(right+B, 20)] on a parse failure, so a parse
                failure costs the harness exactly what it costs the
                baseline and cannot flatter either one

What it adds is context: the harness sections appended after the
telemetry, and the addendum appended to the system prompt that explains
them. With no module on, both are empty and the request equals the
baseline's.
"""

from harness import prompts
from harness import settings as S
from openai_agent import OpenAIMarioAgent, PlanError


class ReasoningModule:

    def __init__(self, llm):
        self.llm = llm
        if S.any_module_on():
            self.system = prompts.BASELINE_SYSTEM_PROMPT + prompts.harness_addendum(
                perception=S.PERCEPTION, scaffold=S.SCAFFOLD, memory=S.MEMORY,
                reflection=S.REFLECTION != "off", lessons=S.LESSONS != "off",
                motion_frames=S.MOTION_FRAMES)
        else:
            self.system = prompts.BASELINE_SYSTEM_PROMPT

    @staticmethod
    def build_text(state: dict, percept, memory_sections: list) -> str:
        text = OpenAIMarioAgent.build_telemetry(state)
        sections = []
        if percept.ram_text:
            sections.append(("PERCEPTION (read from game memory -- exact)", percept.ram_text))
        if percept.vision_text:
            sections.append(("VISUAL ANALYSIS (from a perception model)", percept.vision_text))
        sections += memory_sections
        for title, body in sections:
            text += f"\n\n=== {title} ===\n{body}"
        return text

    def decide(self, state: dict, percept, memory_sections: list) -> dict:
        text = self.build_text(state, percept, memory_sections)
        if S.PRINT_PROMPTS:
            print("----- reasoning prompt -----\n" + text + "\n----------------------------")
        raw = self.llm.complete("reasoning", self.system, text, images=percept.images,
                                json_mode=True)
        parse_failure = False
        try:
            plan, note = OpenAIMarioAgent.parse_plan(raw)
        except PlanError as exc:
            print(f"  [plan] {exc} -- falling back to right+B for 20 frames")
            plan, note = [(3, 20)], f"PARSE FAILURE: {exc}"
            parse_failure = True
        return {"plan": plan, "note": note, "raw": raw, "text": text,
                "parse_failure": parse_failure}
