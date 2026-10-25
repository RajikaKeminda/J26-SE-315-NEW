"""
categorize.py

The guard and the engine explain every blocked or changed action in free text (src/safety/guards.py, src/inference/engines.py).
This maps each note to a stable code so the history can be filtered and counted ("how often did the node-memory check stop a scale-up?").
A reason can hold several notes joined by "; ". Anything not recognised is "other" - the full text is stored next to the codes.
"""

import re

# (code, regex matched at the start of a note) - one entry per message the code base can produce
_RULES = [
    ("clamp", r"clamped replicas"),
    ("no_demand", r"scale-up blocked: no demand"),
    ("busy", r"scale-down blocked: service is busy"),
    ("memory_floor", r"memory floor"),
    ("cooldown", r"scale-down cooldown"),
    ("stabilisation", r"scale-down blocked after a scale-up"),
    ("young_pod", r"scale-down blocked: newest pod"),
    ("resize_cooldown", r"resize cooldown"),
    ("budget", r"blocked: would exceed cluster budget"),
    ("node_memory", r"scale-up blocked: node memory"),
    ("resize_disabled", r"resize disabled"),
    ("observed_only", r"observed only"),
    ("k8s_error", r"k8s error"),
    ("hold", r"HOLD"),
]
_COMPILED = [(code, re.compile(rx)) for code, rx in _RULES]


def categorize(reason):
    """'scale-down cooldown (1 of 3); memory floor 256 MiB ...' -> ['cooldown', 'memory_floor']; None/'' -> []."""
    if not reason:
        return []
    codes = []
    for note in str(reason).split("; "):
        note = note.strip()
        if not note:
            continue
        code = next((c for c, rx in _COMPILED if rx.match(note)), "other")
        if code not in codes:
            codes.append(code)
    return codes
