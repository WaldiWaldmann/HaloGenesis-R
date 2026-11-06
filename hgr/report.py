"""Gemeinsamer Ergebnis-Recorder und Markdown-Report für die Validierungspakete (ab P2)."""
from __future__ import annotations

import json
from pathlib import Path


class Recorder:
    def __init__(self) -> None:
        self.results: dict[str, dict] = {}

    def add(self, test: str, name: str, value: float, threshold: float | None = None,
            passed: bool | None = None, note: str = "") -> None:
        self.results.setdefault(test, {})[name] = {"value": float(value), "threshold": threshold,
                                                   "passed": None if passed is None else bool(passed),
                                                   "note": note}

    def n_failed(self) -> int:
        return sum(1 for t in self.results.values() for r in t.values() if r["passed"] is False)

    def write(self, out: Path, stage: str, title: str, intro: list[str]) -> None:
        nf = self.n_failed()
        lines = [f"# {stage} – {title}", "",
                 f"**Gesamtergebnis:** {'BESTANDEN' if nf == 0 else f'{nf} Test(s) NICHT bestanden'}", ""]
        lines += intro + [""]
        for test, rows in self.results.items():
            lines += [f"## {test}", "", "| Größe | Wert | Schwelle | Status | Anmerkung |", "|---|---|---|---|---|"]
            for name, r in rows.items():
                st = {True: "✅", False: "❌", None: "info"}[r["passed"]]
                thr = "—" if r["threshold"] is None else f"{r['threshold']:.3g}"
                lines.append(f"| {name} | {r['value']:.4g} | {thr} | {st} | {r['note']} |")
            lines.append("")
        (out / f"{stage}_report.md").write_text("\n".join(lines), encoding="utf-8")
        (out / f"{stage}_values.json").write_text(json.dumps(self.results, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
