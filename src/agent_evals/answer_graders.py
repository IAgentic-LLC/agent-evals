"""Graders that read what an answer says. Plain code, so the same answer always
gets the same verdict, and every verdict can be debugged. This is the fragile rung
of the grader ladder: text varies in ways the code did not expect.

`asks_for_known_info` flags an answer that asks the customer to identify their
account when the ticket record already carries a customer ID. It comes in
versions, each one my attempt to fix the errors of the one before, and each is
measured against hand labels (see `grader_check.py`).
"""

import re
from collections.abc import Callable

from agent_evals.schema import EvalCase, Trace

Grader = Callable[[EvalCase, Trace], bool]

_REQUEST = (
    r"(?:provide|share|send|give|confirm|reply with|let (?:us|me) know"
    r"|tell (?:us|me)|enter)"
)
_IDENTITY = (
    r"(?:customer id|account id|account number|user id|username"
    r"|account email|email address|email associated)"
)
# Spanish and French, the two other languages in the test set.
_REQUEST_MORE = (
    _REQUEST[:-1]
    + r"|proporcion\w+|podr[ií]as|indiqu\w+|fournir|pourriez-vous|communiquer)"
)
_IDENTITY_MORE = (
    _IDENTITY[:-1]
    + r"|id de cliente|identificador de cliente|identifiant client"
    + r"|num[eé]ro de compte|n[uú]mero de cuenta|correo electr[oó]nico"
    + r"|adresse e-?mail)"
)


def _pattern(request: str, identity: str, either_order: bool = False) -> re.Pattern:
    forward = rf"{request}\b[^.?!\n]{{0,80}}?{identity}"
    if not either_order:
        return re.compile(forward, re.IGNORECASE)
    backward = rf"{identity}[^.?!\n]{{0,80}}?{request}\b"
    return re.compile(f"{forward}|{backward}", re.IGNORECASE)


_V2 = _pattern(_REQUEST, _IDENTITY)
_V4 = _pattern(_REQUEST_MORE, _IDENTITY_MORE)
_V5 = _pattern(_REQUEST_MORE, _IDENTITY_MORE, either_order=True)


def _used_the_customer_id(trace: Trace) -> bool:
    return any("customer_id" in action for action in trace.actions_taken)


def asks_v1(case: EvalCase, trace: Trace) -> bool:
    """Any mention of the phrase."""
    return "customer id" in trace.answer.lower()


def asks_v2(case: EvalCase, trace: Trace) -> bool:
    """A request word followed, in the same sentence, by an identity word."""
    return bool(_V2.search(trace.answer))


def asks_v3(case: EvalCase, trace: Trace) -> bool:
    """As v2, but only if the specialist never used the ticket's customer ID."""
    return asks_v2(case, trace) and not _used_the_customer_id(trace)


def asks_v4(case: EvalCase, trace: Trace) -> bool:
    """As v3, with Spanish and French request and identity words added."""
    return bool(_V4.search(trace.answer)) and not _used_the_customer_id(trace)


def asks_v5(case: EvalCase, trace: Trace) -> bool:
    """As v4, and also an identity word followed by a request word."""
    return bool(_V5.search(trace.answer)) and not _used_the_customer_id(trace)


ASKS_FOR_KNOWN_INFO: dict[str, Grader] = {
    "v1": asks_v1,
    "v2": asks_v2,
    "v3": asks_v3,
    "v4": asks_v4,
    "v5": asks_v5,
}


def acted_on_the_right_customer(case: EvalCase, trace: Trace) -> bool:
    """True if every action that names a customer names the ticket's customer.

    This checks a tool argument, not prose, so it does not depend on wording or
    language.
    """
    ticket_customer = case.input.get("customer_id")
    return all(
        action.get("customer_id", ticket_customer) == ticket_customer
        for action in trace.actions_taken
    )


GRADERS: dict[str, dict[str, Grader]] = {"asks_for_known_info": ASKS_FOR_KNOWN_INFO}
