#!/usr/bin/env python3
"""
Stimuli writer for Self–Other Epistemic Audit.

v2 (harder): multi-step / subtle planted errors live in stimuli.json (hand-crafted).
v1 easy backup: stimuli_v1_easy.json

This script validates schema and can optionally restore v2 from the embedded HARD set
if stimuli.json is missing. Prefer editing stimuli.json directly for quality control.

Sprint: Apart Research AI x Epistemics, 13–15 Nov 2026.
"""

from __future__ import annotations

import json
from pathlib import Path

DIR = Path(__file__).resolve().parent
OUT = DIR / "stimuli.json"
REQUIRED = ("id", "pregunta", "respuesta_incorrecta_plausible", "error_real", "categoria")


def main() -> None:
    if not OUT.exists():
        raise SystemExit(f"Missing {OUT}. Restore from stimuli_v1_easy.json or recreate harder set.")
    items = json.loads(OUT.read_text(encoding="utf-8"))
    assert len(items) >= 20, f"need ≥20 stimuli, got {len(items)}"
    ids = set()
    for it in items:
        for k in REQUIRED:
            assert k in it and str(it[k]).strip(), f"missing {k} in {it.get('id')}"
        assert it["id"] not in ids, f"duplicate id {it['id']}"
        ids.add(it["id"])
    print(f"OK: {len(items)} stimuli validated → {OUT}")
    print(f"ids: {', '.join(sorted(ids))}")


if __name__ == "__main__":
    main()
