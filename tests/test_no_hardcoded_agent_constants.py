"""Extends the earlier ID-grep check to numeric
literals -- no building-specific id/coordinate, and no numeric behaviour/
physics constant, may live in dms/ code outside a config loader module
(dms/scenario/agent_defaults.py, dms/viz/viz_config.py). This scans the
walker/report modules, which are now fully config-driven,
and fails on any numeric literal that isn't on the reviewed allowlist below
-- every surviving entry is pure mathematics (a geometric identity, a unit
conversion, or a numerical-stability epsilon), never a tunable parameter.
`dms/scenario/population.py`'s persona walking speeds and the PARTIAL-state
door factor (both previously allowlisted as "inherited, out of scope") were
moved to configs/agent_defaults.yaml in this same follow-up and no longer
appear here at all.

Older modules not touched here (dms/graph/
builder.py, dms/fields/eikonal.py's own pre-existing constants, etc.) are
still out of scope here -- retrofitting every already-tested, already-cited
constant in the whole repository is a separate, much larger undertaking
this does not attempt."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Fully config-driven -- every number here was
# individually reviewed.
SCANNED_FILES = [
    "dms/scenario/evacuation.py",
    "dms/geometry/body_class_report.py",
    "dms/scenario/population.py",
]

# (file, literal value) -> why it's not a tunable behaviour/physics constant.
# Narrowed to pure maths only --
# the PARTIAL-state door factor (was a bare 0.5/1.0 pair) now reads
# SYSTEM_SPEC's own omega_p from config (dms/scenario/evacuation.py's
# `agent_defaults.partial_capacity_omega_p`), and the outdoor "radius
# doesn't apply here" case is an explicit `None`, not a dummy 0.0 float.
ALLOWLIST: dict[tuple[str, float], str] = {
    ("dms/scenario/evacuation.py", 2.0): "diameter/segment-midpoint math (2*radius, 2*max_radius, /2.0 midpoints) -- a geometric identity, not a tunable parameter",
    ("dms/scenario/evacuation.py", 0.5): "the geometric midpoint fraction between an agent and a look-ahead candidate (0.5 * look_indoor_m) -- structurally 'half', not a design choice",
    ("dms/scenario/evacuation.py", 1.0): "1 / stair.Q and 1 / omega_p-adjacent unit math -- persons-per-second -> seconds-per-person, a mathematical identity",
    ("dms/scenario/evacuation.py", 0.0): "additive identity / zero-initializer (t=0 start) -- not a tunable constant",
    ("dms/scenario/evacuation.py", 1e-9): "division-by-zero numerical-stability epsilon, same class as eikonal.py's pre-existing 1e-12/1e-6 epsilons",
    ("dms/geometry/body_class_report.py", 2.0): "diameter = 2 * radius, the same geometric identity the door-passability formula (2r + margin) is defined by",
    ("dms/scenario/population.py", 1.0): "two distinct, reviewed uses: DEFAULT_BODY_CLASS_MIX's {'standard': 1.0} (100% probability mass, a structural identity, not a tunable share) and sample_population's rescale factor when N is not given (scale=1.0 means 'no rescaling', a mathematical identity)",
}


def _iter_numeric_literals(tree: ast.Module):
    """Float literals only -- bare int literals in this codebase are
    overwhelmingly structural (slice/index bounds, `k=1` sample counts,
    `range(n)` state-machine sizes, retry-attempt caps), while every
    genuine tunable behaviour/physics constant here is written as a float
    (`0.3`, `1.2`, `7.6`, ...), matching this repo's own existing style. An
    int masquerading as a tunable constant would still be unusual enough to
    warrant catching by eye in review, not by this blunt a scan."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            yield node.lineno, node.value


def test_no_unreviewed_numeric_literals_in_fully_config_driven_modules():
    problems = []
    for rel in SCANNED_FILES:
        path = REPO_ROOT / rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for lineno, value in _iter_numeric_literals(tree):
            if (rel, value) in ALLOWLIST:
                continue
            problems.append(f"{rel}:{lineno}: unreviewed numeric literal {value!r}")
    assert problems == [], "\n".join(problems)
