"""Labels from people and from judges, side by side (chapter 17).

A label is `supported` or `unsupported` per item. My readings have three classes, so a
convention says where `borderline` goes: `strict` counts it as unsupported, `lenient` as
supported. A judge's label is its first-pass verdict. Adjudication records the changes I
made to my own readings after seeing where judges disagreed with me.
"""

import json
from pathlib import Path
from typing import Any

from agent_evals import agreement
from agent_evals.judge import load_items
from agent_evals.judge_report import load_rows

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ROOT / "datasets" / "judge_items_v1.jsonl"
ADJUDICATION = ROOT / "datasets" / "judge_adjudication_v1.jsonl"
SAMPLE = ROOT / "datasets" / "human_label_sample_v1.jsonl"


def load_adjudication(path: Path = ADJUDICATION) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = [
        json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()
    ]
    return {r["item_id"]: r for r in rows}


def author_labels(
    items: list[dict[str, Any]],
    convention: str = "strict",
    adjudicated: bool = False,
) -> dict[str, str]:
    """My label for each item. Planted faults are unsupported by construction."""
    changes = load_adjudication() if adjudicated else {}
    out = {}
    for item in items:
        reading = changes.get(item["item_id"], {}).get("decision", item["reading"])
        if reading == "planted":
            reading = "stretch"
        bad = ("borderline", "stretch") if convention == "strict" else ("stretch",)
        out[item["item_id"]] = "unsupported" if reading in bad else "supported"
    return out


def judge_labels(run: str | Path) -> dict[str, str]:
    """A judge's first-pass verdict for each item it judged."""
    path = Path(run)
    if not path.exists():
        path = ROOT / "runs" / str(run)
    return {
        r["item_id"]: r["verdict"]
        for r in load_rows([path])
        if r["pass"] == 1 and r["verdict"]
    }


def second_pass_labels(run: str | Path) -> dict[str, str]:
    path = Path(run)
    if not path.exists():
        path = ROOT / "runs" / str(run)
    return {
        r["item_id"]: r["verdict"]
        for r in load_rows([path])
        if r["pass"] == 2 and r["verdict"]
    }


def _row(name: str, a: list[str], b: list[str], groups: list[str]) -> str:
    kappa = agreement.cohen_kappa(a, b)
    low, high = agreement.bootstrap_interval(agreement.cohen_kappa, a, b, groups=groups)
    shown = f"{kappa:.2f} ({low:.2f} to {high:.2f})"
    return (
        f"{name:<16}{len(a):>6}{agreement.percent_agreement(a, b):>8.3f}{shown:>24}"
        f"{agreement.pabak(a, b):>8.2f}"
    )


def render_matrix(
    items: list[dict[str, Any]],
    raters: dict[str, dict[str, str]],
    reference: dict[str, str],
    group: str = "real",
    reference_name: str = "author",
) -> str:
    """Each rater's agreement with a reference labeling, on the real items or all of them."""
    kept = [i for i in items if group == "all" or i["group"] == group]
    lines = [f"agreement with {reference_name}"]
    lines.append(
        f"{'rater':<16}{'items':>6}{'agree':>8}{'kappa (95% interval)':>24}{'PABAK':>8}"
    )
    for name, labels in raters.items():
        ids = [i for i in kept if i["item_id"] in labels]
        a = [reference[i["item_id"]] for i in ids]
        b = [labels[i["item_id"]] for i in ids]
        lines.append(_row(name, a, b, [i["question"] for i in ids]))
    return "\n".join(lines) + "\n"


def render_retest(items: list[dict[str, Any]], runs: list[str]) -> str:
    """A judge against itself: pass 1 against pass 2, on every item."""
    lines = [f"{'judge run':<16}{'items':>6}{'agree':>8}{'kappa':>8}"]
    for run in runs:
        one, two = judge_labels(run), second_pass_labels(run)
        ids = [
            i["item_id"] for i in items if i["item_id"] in one and i["item_id"] in two
        ]
        a, b = [one[i] for i in ids], [two[i] for i in ids]
        lines.append(
            f"{run.replace('judge-', ''):<16}{len(ids):>6}"
            f"{agreement.percent_agreement(a, b):>8.3f}{agreement.cohen_kappa(a, b):>8.2f}"
        )
    return "\n".join(lines) + "\n"


def render_calibration(
    items: list[dict[str, Any]], truth: dict[str, str], judge: dict[str, str], name: str
) -> str:
    """Correct a judge's flag rate using half the questions, and check it on the other half."""
    real = [i for i in items if i["group"] == "real"]
    half = [agreement.group_of(i["question"]) for i in real]
    cal = [i for i, h in zip(real, half) if h == 0]
    test = [i for i, h in zip(real, half) if h == 1]

    def flags(rows):
        return [judge[i["item_id"]] == "unsupported" for i in rows]

    def bad(rows):
        return [truth[i["item_id"]] == "unsupported" for i in rows]

    lines = [
        f"{'':<34}{'items':>6}{'bad answers':>13}{'judge flags':>13}",
        f"{'calibration questions':<34}{len(cal):>6}{sum(bad(cal)):>13}{sum(flags(cal)):>13}",
        f"{'evaluation questions':<34}{len(test):>6}{sum(bad(test)):>13}{sum(flags(test)):>13}",
        "",
    ]
    if not any(bad(cal)):
        lines.append(
            "no bad answer in the calibration questions: sensitivity is undefined"
        )
        return "\n".join(lines) + "\n"
    sens, spec = agreement.sensitivity_specificity(bad(cal), flags(cal))
    lines.append(
        f"{name}: sensitivity {sens:.2f} and specificity {spec:.2f} on the calibration questions"
    )
    apparent = sum(flags(test)) / len(test)
    lines.append(
        f"rate the judge reports on the evaluation questions: {100 * apparent:.1f}%"
    )
    lines.append(
        f"rate my labels give on the evaluation questions:     {100 * sum(bad(test)) / len(test):.1f}%"
    )
    try:
        point, (low, high) = agreement.calibrated_rate(
            bad(cal),
            flags(cal),
            [i["question"] for i in cal],
            flags(test),
            [i["question"] for i in test],
        )
        lines.append(
            f"corrected rate:                                    {100 * point:.1f}% "
            f"({100 * low:.1f}% to {100 * high:.1f}%)"
        )
    except ZeroDivisionError:
        lines.append(
            "no correction possible: on these labels the judge is no better than a coin"
        )
    return "\n".join(lines) + "\n"


def render_sensitivity(
    items: list[dict[str, Any]], truth: dict[str, str], judge: dict[str, str]
) -> str:
    """Sensitivity and specificity on all the real items, resampling whole questions."""
    real = [i for i in items if i["group"] == "real" and i["item_id"] in judge]
    t = [truth[i["item_id"]] == "unsupported" for i in real]
    f = [judge[i["item_id"]] == "unsupported" for i in real]
    groups = [i["question"] for i in real]
    sens, spec = agreement.sensitivity_specificity(t, f)

    def part(which):
        def stat(truth_, flag_):
            return agreement.sensitivity_specificity(truth_, flag_)[which]

        return stat

    ci = [agreement.bootstrap_interval(part(k), t, f, groups=groups) for k in (0, 1)]
    return (
        f"{'':<14}{'value':>8}{'95% interval':>18}\n"
        f"{'sensitivity':<14}{sens:>8.2f}{ci[0][0]:>10.2f} to {ci[0][1]:.2f}\n"
        f"{'specificity':<14}{spec:>8.2f}{ci[1][0]:>10.2f} to {ci[1][1]:.2f}\n"
        f"bad answers {sum(t)} of {len(t)}, from {len({g for g, x in zip(groups, t) if x})} questions\n"
    )


def render_plan(
    sens: float,
    spec: float,
    prevalence: float,
    sizes: tuple[int, ...],
    tests: tuple[int, ...],
) -> str:
    lines = [
        f"{'calibration items':<20}"
        + "".join(f"{f'{n} test items':>16}" for n in tests)
    ]
    columns = [
        dict(agreement.planning_widths(sens, spec, prevalence, n, sizes)) for n in tests
    ]
    for size in sizes:
        cells = "".join(f"{100 * c[size]:>13.1f} pts" for c in columns)
        lines.append(f"{size:<20}{cells}")
    return "\n".join(lines) + "\n"


def render_second(
    items: list[dict[str, Any]],
    second: dict[str, str],
    author: dict[str, str],
    judge: dict[str, str],
) -> str:
    """A second person's labels against mine and against a judge. `unsure` is set aside."""
    ids = [
        i
        for i in items
        if i["item_id"] in second
        and second[i["item_id"]] in ("supported", "unsupported")
    ]
    unsure = sum(1 for i in items if second.get(i["item_id"]) == "unsure")
    lines = [f"{len(ids)} labeled, {unsure} marked not sure"]
    raters = {"author": author, "judge": judge}
    lines.append(
        f"{'second person vs':<18}{'items':>6}{'agree':>8}{'kappa (95% interval)':>24}{'PABAK':>8}"
    )
    for name, labels in raters.items():
        usable = [i for i in ids if i["item_id"] in labels]
        a = [second[i["item_id"]] for i in usable]
        b = [labels[i["item_id"]] for i in usable]
        lines.append(_row(name, a, b, [i["question"] for i in usable]))
    return "\n".join(lines) + "\n"


def item_list() -> list[dict[str, Any]]:
    return load_items(ITEMS)
