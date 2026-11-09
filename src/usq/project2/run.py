"""Research entry point: factor selection then validation and sealed OOS test."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from . import screen, models


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reuse-factors", help="Optional computed factors, never reused outcome-based choices")
    p.add_argument("--reuse-screen", help="Reuse a complete, same-protocol factor screen and numerical feature cache")
    p.add_argument("--all", action="store_true", help="Also run factor quintiles, Barra-style attribution and the offline result page")
    args = p.parse_args()
    if args.reuse_screen:
        src, dst = Path(args.reuse_screen), Path(args.output)
        if json.loads((src/"research_protocol.json").read_text()) != screen.POLICY:
            raise ValueError("Cannot reuse a different factor-selection protocol")
        selection = json.loads((src/"factor_selection.json").read_text())
        if selection["protocol_sha256"] != hashlib.sha256((src/"research_protocol.json").read_bytes()).hexdigest():
            raise ValueError("Factor screen protocol hash differs")
        dst.mkdir(exist_ok=True, parents=True)
        for sub in ("factors", "events", "design", "evaluation", "barra_style"):
            if (src/sub).exists():
                shutil.copytree(src/sub, dst/sub, dirs_exist_ok=True)
        for name in ("factor_selection.json", "factor_catalog.json", "research_protocol.json"):
            shutil.copy2(src/name, dst/name)
    else:
        screen.run(args.database, args.output, args.reuse_factors)
    models.run(args.database, args.output)
    if args.all:
        from . import diagnostics, risk, report
        diagnostics.run(args.database, args.output)
        risk.run(args.database, args.output)
        report.run(args.database, args.output)
