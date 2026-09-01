"""Standard Comites capability suite — shared contracts for estates with a Magister.

Some Comites.ai deployments add an optional "Magister" (chief-of-staff) agent
that coordinates the other agents: it polls them for opinions and status, and
it is the sole writer to shared assets (todo list, calendars, tracking sheets).
This module carries the STANDARD, cross-agent contracts for that arrangement:

  - Four standard inquiries every agent MAY answer (prune inquiries.json to
    the ones that genuinely fit your agent's domain — the instruction
    fragment below is built from that file, so pruning it prunes the prompt
    in the same commit):
      review_idea    — vote for/against an idea, with 1-10 conviction
      goal_progress  — score the user's recent progress on a goal/mantra, 1-10
      focus_items    — what the user should focus on today/this week
      daily_metric   — report a tracked metric's value for ANY date
  - The shared conviction scale (so scores are comparable across agents).
  - The instruction fragment that teaches an agent these behaviors.

THE WHOLE SUITE IS CONDITIONAL. It activates only when MAGISTER_DISPLAY_NAME
is set in the environment. Unset (the default), `magister_instruction()`
returns "" and register_agent.py skips inquiries marked "standard": true —
a standalone agent behaves exactly like an agent built before this module
existed. Comites and The Forum work with or without a Magister.

Transport note: there is no Python client for agent-to-agent calls. Talking
to the Magister is a PROMPT-level convention — the model calls the Forum's
`query_agent` tool (agents MCP toolset) with the Magister's display name and
a message in these formats. This module only provides the format strings and
lightweight parsers (used mainly by Magister-side code and tests).
"""
import json
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Conviction scale — quoted verbatim in every agent's instruction so a "7"
# means the same thing no matter which agent said it.
# ---------------------------------------------------------------------------
CONVICTION_SCALE = (
    "Conviction scale (1-10, shared by every agent):\n"
    "  1-2  barely worth mentioning / mild preference\n"
    "  3-4  real but minor; fine to ignore this week\n"
    "  5-6  meaningful; should influence the decision\n"
    "  7-8  strong professional opinion; ignoring this has consequences\n"
    "  9-10 near-certain harm/miss if ignored (deadline passes, health "
    "risk, hard commitment broken)\n"
    "Conviction measures strength of opinion, independent of direction."
)

# Universal abstention/absence response. Abstaining is ENCOURAGED: a
# confident opinion inside your domain is worth more than ten polite ones
# outside it. Too much low-value information is its own failure mode.
NO_DATA_PREFIX = "NO_DATA:"


def is_no_data(text: str) -> bool:
    """True when a reply is an abstention/absence (`NO_DATA: <reason>`).

    Case-insensitive, tolerant of leading whitespace. Check this before (or
    alongside) the parsers below — parse_focus_items in particular returns a
    plain list and does NOT distinguish abstention from "FOCUS: none".
    """
    return text.strip()[:len(NO_DATA_PREFIX)].upper() == NO_DATA_PREFIX


# ---------------------------------------------------------------------------
# The four standard contracts. These strings are the single source of truth:
# inquiries.json must carry them verbatim (register_agent.py hard-fails the
# deploy on drift, and tests/test_comites_standard.py checks the shipped
# stub), and magister_instruction() quotes them from here.
# ---------------------------------------------------------------------------
STANDARD_CONTRACTS = {
    "review_idea": {
        "request_format": (
            "AGENT_QUERY: review_idea | idea=<free text> | context=<optional>"
        ),
        "response_format": (
            "IDEA_REVIEW: verdict=<for|against|abstain> conviction=<1-10> "
            "| reason=<one sentence>"
        ),
        "guidance": (
            "  Conviction measures strength of opinion, independent of the "
            "for/against direction."
        ),
    },
    "goal_progress": {
        "request_format": (
            "AGENT_QUERY: goal_progress | item=<exact goal or mantra string> "
            "| window_days=<n>"
        ),
        "response_format": (
            "GOAL_PROGRESS: item=<string> score=<1-10> | evidence=<one sentence>"
        ),
        "guidance": (
            "  Score from your own records (memory, logs, tracked data), "
            "not vibes."
        ),
    },
    "focus_items": {
        "request_format": (
            "AGENT_QUERY: focus_items | horizon=<today|week> | today=<YYYY-MM-DD> "
            "| context=<optional trip/calendar hints>"
        ),
        "response_format": (
            "FOCUS: none — or 1-3 lines of: "
            "FOCUS: item=<action> conviction=<1-10> | why=<deadline or goal linkage>"
        ),
        "guidance": (
            "  Conviction here = urgency (deadline proximity) x importance "
            "to the user's goals. Trust the `today=` date in the request."
        ),
    },
    "daily_metric": {
        "request_format": (
            "AGENT_QUERY: daily_metric | metric=<column key> | date=<YYYY-MM-DD>"
        ),
        "response_format": (
            "METRIC: metric=<key> date=<YYYY-MM-DD> value=<number|text> "
            "— or NO_DATA: <reason>"
        ),
        "guidance": (
            "  Must work for ANY date you have data for, not just today."
        ),
    },
}


def magister_display_name() -> str:
    """The configured Magister's display name, or "" when there is none."""
    return os.environ.get("MAGISTER_DISPLAY_NAME", "").strip()


def published_standard_inquiries() -> list:
    """Names of standard inquiries this agent's inquiries.json still carries.

    inquiries.json ships in the deploy bundle precisely so this module can
    read it: pruning the file prunes the instruction fragment in the same
    commit (contract atomicity). Returns [] when the file is absent or
    unreadable — the fragment then teaches no standard inquiries but keeps
    the write-via-Magister rule.
    """
    try:
        with open(Path(__file__).parent / "inquiries.json", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    return [
        inq.get("name") for inq in data.get("inquiries", [])
        if inq.get("standard") and inq.get("name") in STANDARD_CONTRACTS
    ]


def magister_instruction() -> str:
    """Instruction fragment for estates with a Magister; "" without one.

    Concatenate onto the agent's instruction:
        instruction=YOUR_INSTRUCTION + magister_instruction()

    With MAGISTER_DISPLAY_NAME unset this is a no-op, so the same agent.py
    works in deployments with and without a Magister. With it set, the
    fragment teaches only the standard inquiries actually present in this
    agent's inquiries.json.
    """
    magister = magister_display_name()
    if not magister:
        return ""

    parts = [
        "\n\n## Working with the Magister (chief of staff)\n\n"
        f"This deployment has a coordinating agent, \"{magister}\". Two "
        "standing duties:\n\n"
        f"**1. Shared assets are written ONLY by {magister}.** Never write "
        "directly to the user's todo list, calendar, email, or shared "
        "tracking documents. To add or change a todo, use the `query_agent` "
        f"tool (agents MCP toolset) with agent_name=\"{magister}\" and a "
        "message in the format published in their register (see "
        "`get_agent_inquiries`). Your own domain assets remain yours."
    ]

    published = published_standard_inquiries()
    if published:
        lines = [
            f"\n\n**2. Answer {magister}'s standard inquiries honestly and "
            "from your own records.** Requests arrive prefixed "
            "`[From Agent: ... | On Behalf Of: <user>]` in these formats:"
        ]
        for name in published:
            contract = STANDARD_CONTRACTS[name]
            lines.append(f"\n- {contract['request_format']}\n"
                         f"  Reply: {contract['response_format']}\n"
                         f"{contract['guidance']}")
        parts.append("".join(lines))

    parts.append(
        f"\n\n{CONVICTION_SCALE}\n\n"
        f"Abstention is encouraged, not just allowed: reply `{NO_DATA_PREFIX} "
        "<reason>` for anything outside your domain or absent from your "
        "records. When in doubt, abstain — a confident opinion inside your "
        "domain is worth more than ten polite ones outside it. The standard "
        "inquiries listed above are the only ones you answer; anything else "
        f"gets `{NO_DATA_PREFIX} not an inquiry I publish`."
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Parsers — used by Magister-side code (and tests) to read standard replies.
# Free-text tolerant: they extract what they can and return {} / [] on no
# match. Each parser returns one of TWO shapes; see its docstring. Conviction
# and score values are clamped into 1-10.
# ---------------------------------------------------------------------------
def _clamp(value: int) -> int:
    return max(1, min(10, value))


def parse_review_idea(text: str) -> dict:
    """Parse an IDEA_REVIEW reply.

    Returns {verdict, conviction, reason} on a match, {verdict: "abstain",
    conviction: None, no_data} on a NO_DATA reply, and {} when unparseable.
    """
    if is_no_data(text):
        return {"verdict": "abstain", "conviction": None, "no_data": text.strip()}
    m = re.search(
        r"IDEA_REVIEW:\s*verdict=(for|against|abstain)\s+conviction=(\d+)"
        r"(?:\s*\|\s*reason=([^\n]*))?",
        text, re.IGNORECASE,
    )
    if not m:
        return {}
    return {
        "verdict": m.group(1).lower(),
        "conviction": _clamp(int(m.group(2))),
        "reason": (m.group(3) or "").strip(),
    }


def parse_goal_progress(text: str) -> dict:
    """Parse a GOAL_PROGRESS reply.

    Returns {item, score, evidence} on a match, {no_data} on a NO_DATA
    reply, and {} when unparseable.
    """
    if is_no_data(text):
        return {"no_data": text.strip()}
    m = re.search(
        r"GOAL_PROGRESS:\s*item=(.+?)\s+score=(\d+)(?:\s*\|\s*evidence=([^\n]*))?",
        text, re.IGNORECASE,
    )
    if not m:
        return {}
    return {
        "item": m.group(1).strip(),
        "score": _clamp(int(m.group(2))),
        "evidence": (m.group(3) or "").strip(),
    }


def parse_focus_items(text: str) -> list:
    """Parse a FOCUS reply -> list of {item, conviction, why} (may be []).

    An empty list means "no focus items" AND ALSO results from a NO_DATA
    abstention — callers that care about the difference must check
    `is_no_data(text)` themselves before calling.
    """
    items = []
    for m in re.finditer(
        r"FOCUS:\s*item=(.+?)\s+conviction=(\d+)(?:\s*\|\s*why=([^\n]*))?",
        text, re.IGNORECASE,
    ):
        items.append({
            "item": m.group(1).strip(),
            "conviction": _clamp(int(m.group(2))),
            "why": (m.group(3) or "").strip(),
        })
    return items


def parse_daily_metric(text: str) -> dict:
    """Parse a METRIC reply.

    Returns {metric, date, value} on a match, {no_data} on a NO_DATA reply,
    and {} when unparseable.
    """
    if is_no_data(text):
        return {"no_data": text.strip()}
    m = re.search(
        r"METRIC:\s*metric=(\S+)\s+date=(\d{4}-\d{2}-\d{2})\s+value=([^\n]+)",
        text, re.IGNORECASE,
    )
    if not m:
        return {}
    return {
        "metric": m.group(1).strip(),
        "date": m.group(2),
        "value": m.group(3).strip(),
    }
