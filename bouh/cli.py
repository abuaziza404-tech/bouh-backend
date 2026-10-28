"""CLI worker entry point. GEE credentials come from runtime/ADC, never JSON."""
import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="BOUH auditable GEE worker")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("run"))
    parser.add_argument("--submit", action="store_true", help="Start asynchronous Drive exports")
    parser.add_argument("--status", action="store_true", help="Refresh task ids in an existing run")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output/"run_manifest.json"
    import ee
    from .config import Config
    raw = json.loads(args.config.read_text())
    unknown = set(raw)-{f.name for f in fields(Config)}
    if unknown:
        parser.error(f"Unknown configuration keys: {sorted(unknown)}")
    cfg = Config(**raw).validate()
    logging.basicConfig(level=logging.INFO)
    report = {"state": "INITIALIZING", "config": cfg.record(),
              "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        ee.Initialize(project=cfg.project)
        if args.status:
            report = json.loads(report_path.read_text())
            for name, record in report.get("exports", {}).items():
                if record.get("id"):
                    report["exports"][name] = ee.data.getTaskStatus(record["id"])[0]
        else:
            if report_path.exists():
                report = json.loads(report_path.read_text())
                logging.error("Run exists; use --status or a new --output directory")
                return 1
            from .pipeline import process
            from .exporting import build_exports, submit_exports
            product = process(cfg, report=report)
            report = product.report
            jobs = build_exports(product, cfg)
            report["export_plan"] = list(jobs)
            # Retain serializable image/vector graphs for audit, NOT raw data.
            (args.output/"evidence_graph.json").write_text(product.evidence.serialize())
            (args.output/"target_graph.json").write_text(product.targets.serialize())
            if args.submit or cfg.start_exports:
                submit_exports(jobs, report)
            else:
                report["exports"] = {name: {"state": "NOT_SUBMITTED"} for name in jobs}
    except Exception as exc:
        report.update(state="BLOCKED_OR_FAILED", error_type=type(exc).__name__, error=str(exc))
        logging.exception("BOUH stopped without emitting a trusted result")
        return 1
    finally:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        # Hash only actual local bytes. Do not attribute these to source rasters.
        paths = sorted(p for p in args.output.iterdir() if p.is_file() and p.name != "CHECKSUMS.sha256")
        (args.output/"CHECKSUMS.sha256").write_text("".join(
            hashlib.sha256(p.read_bytes()).hexdigest()+"  "+p.name+"\n" for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
