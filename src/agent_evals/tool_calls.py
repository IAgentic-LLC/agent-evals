"""Grade the tool calls themselves: was each call well formed, allowed and in order?

This is separate from what the calls changed (chapters 6 to 8) and from whether the
agent should have acted at all. A call can be perfectly formed and still be useless,
so the same records also say whether a call came back empty.
"""

import json
from typing import Any

from jsonschema import Draft202012Validator
from triage_app.handoff import HANDOFF_TOOL
from triage_app.tools import (
    _RUNBOOKS,
    BILLING_TOOLS,
    SECURITY_TOOLS,
    TECHNICAL_TOOLS,
)

from agent_evals.schema import EvalCase, Trace

RULES = (
    "unknown_tool",
    "tool_not_offered",
    "invalid_arguments",
    "wrong_customer",
    "refund_without_lookup",
    "refund_above_invoice",
)

# The schema each tool declares to the model, made strict: an argument the schema
# does not name is a fault, so a typo in a key cannot pass.
SCHEMAS: dict[str, dict[str, Any]] = {}
for _tool in BILLING_TOOLS + TECHNICAL_TOOLS + SECURITY_TOOLS + [HANDOFF_TOOL]:
    _fn = _tool["function"]
    SCHEMAS[_fn["name"]] = {**_fn["parameters"], "additionalProperties": False}


def _json(text: str | None) -> dict[str, Any]:
    try:
        value = json.loads(text or "")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def call_problems(case: EvalCase, trace: Trace) -> list[tuple[str, str]]:
    """The rules the run's tool calls broke, as (rule, tool name) pairs."""
    found: list[tuple[str, str]] = []
    invoices: dict[str, float] = {}
    ticket_customer = case.input.get("customer_id")
    for call in trace.tool_calls:
        name, args = call["name"], call["arguments"]
        if name not in SCHEMAS:
            found.append(("unknown_tool", name))
            continue
        if not call["offered"]:
            found.append(("tool_not_offered", name))
        if list(Draft202012Validator(SCHEMAS[name]).iter_errors(args)):
            found.append(("invalid_arguments", name))
        customer = args.get("customer_id")
        if customer is not None and customer != ticket_customer:
            found.append(("wrong_customer", name))
        if name == "look_up_invoice":
            reply = _json(call["result"])
            invoices[customer] = float(reply.get("amount_usd", 0.0))
        if name == "issue_refund":
            if customer not in invoices:
                found.append(("refund_without_lookup", name))
            elif args.get("amount_usd", 0) > invoices[customer]:
                found.append(("refund_above_invoice", name))
    return found


def is_empty(call: dict[str, Any]) -> bool | None:
    """Did the call come back with nothing to use? None when there is no result."""
    if call["result"] is None:
        return None
    reply = _json(call["result"])
    return "error" in reply or reply.get("result") == "no matching runbook entry"


# The runbook entry that helps with a ticket, by the ticket's `runbook_topic` slice.
# A ticket with any other topic has no entry that helps.
RUNBOOK_ENTRY = {"crash": _RUNBOOKS["app crash"], "login": _RUNBOOKS["login"]}
OUTCOMES = ("relevant", "irrelevant", "empty")


def search_outcome(case: EvalCase, call: dict[str, Any]) -> str | None:
    """`relevant`, `irrelevant` or `empty` for a runbook search that has a result.

    An entry that comes back is irrelevant when it is not the one for this ticket's
    topic. Only a relevant entry counts as a useful search.
    """
    if call["name"] != "search_runbook" or call["result"] is None:
        return None
    if is_empty(call):
        return "empty"
    entry = RUNBOOK_ENTRY.get(case.slices.get("runbook_topic", ""))
    return "relevant" if _json(call["result"]).get("result") == entry else "irrelevant"


def render(run: str, cases: dict[str, EvalCase], traces: list[Trace]) -> str:
    """A table of calls per tool, then how often each rule was broken."""
    calls, wrong, empty, seen = {}, {}, {}, {}
    rule_hits = dict.fromkeys(RULES, 0)
    for t in traces:
        bad_tools = call_problems(cases[t.case_id], t)
        for rule, _ in bad_tools:
            rule_hits[rule] += 1
        for call in t.tool_calls:
            name = call["name"]
            calls[name] = calls.get(name, 0) + 1
            gap = is_empty(call)
            if gap is not None:
                seen[name] = seen.get(name, 0) + 1
                empty[name] = empty.get(name, 0) + bool(gap)
        for _, name in set(bad_tools):
            wrong[name] = wrong.get(name, 0) + 1
    lines = [
        f"Tool calls in {run} ({len(traces)} traces, {sum(calls.values())} calls)",
        "",
        "tool                 calls  broke a rule  empty result",
    ]
    for name in sorted(calls):
        gap = f"{empty.get(name, 0)} of {seen[name]}" if name in seen else "no result"
        lines.append(f"{name:<20}{calls[name]:>6}{wrong.get(name, 0):>14}  {gap:>12}")
    outcomes = dict.fromkeys(OUTCOMES, 0)
    for t in traces:
        for call in t.tool_calls:
            kind = search_outcome(cases[t.case_id], call)
            if kind:
                outcomes[kind] += 1
    if any(outcomes.values()):
        widths = [len(k) + 2 for k in OUTCOMES]
        label = "search_runbook results"
        lines += [""]
        lines.append(
            f"{label:<24}" + "".join(f"{k:>{w}}" for k, w in zip(OUTCOMES, widths))
        )
        lines.append(
            " " * 24 + "".join(f"{outcomes[k]:>{w}}" for k, w in zip(OUTCOMES, widths))
        )
    lines += ["", "rule                    calls that broke it"]
    lines += [f"{rule:<24}{hits:>5}" for rule, hits in rule_hits.items()]
    return "\n".join(lines) + "\n"
