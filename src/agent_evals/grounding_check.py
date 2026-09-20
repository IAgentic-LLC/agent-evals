"""Show that the grounding checks can fail (chapter 13).

A scripted model answers every question from the retrieved text, and can be told to go
wrong in exactly one way. The faithful script must raise no flag at all. Each faulty
script must raise the flag aimed at it on every answer where that flag applies. No key
and no network are needed: retrieval comes from the recorded embeddings.
"""

import asyncio
from pathlib import Path

from agent_evals import grounding
from agent_evals.adapters.pkgintel import PkgAnswerAdapter, ScriptedRagClient
from agent_evals.runner import load_cases, run_cases

DATASET = Path(__file__).resolve().parents[2] / "datasets" / "pkg_answers_v1.jsonl"
# Each faulty script and the flag aimed at it.
FAULTS = {
    "cites_unretrieved": "invalid_citation",
    "names_outside": "outside_name",
    "states_version": "new_number",
    "cites_nothing": "no_citation",
    "ignores_context": "ignored_context",
    "invalid_json": "error",
}


async def _run(mode: str, cases, names):
    adapter = PkgAnswerAdapter(client=ScriptedRagClient(mode, names))
    return await run_cases(cases, adapter, concurrency=1)


def main(names: list[str]) -> int:
    cases = load_cases(DATASET)
    by_id = {c.case_id: c for c in cases}
    print(f"{'script':<19}{'flag aimed at it':<18}{'caught':>10}{'other flags':>13}")
    wrong = False
    for mode in ("faithful", *FAULTS):
        aimed = FAULTS.get(mode)
        traces = asyncio.run(_run(mode, cases, names))
        applies = caught = other = 0
        for trace in traces:
            found = grounding.check(by_id[trace.case_id], trace, names)
            raised = {f for f in grounding.FLAGS if found[f]}
            if aimed and found[aimed] is not None:
                applies += 1
                caught += aimed in raised
            other += len(raised - {aimed})
        if aimed:
            cell = f"{caught} of {applies}"
            wrong = wrong or applies == 0 or caught != applies
        else:
            cell = "-"
        wrong = wrong or (not aimed and other > 0)
        print(f"{mode:<19}{aimed or '(none)':<18}{cell:>10}{other:>13}")
    return 1 if wrong else 0
