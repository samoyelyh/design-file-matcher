"""Read-only production inventory and recoverable SQLite snapshot for rollout."""
import argparse
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--before", action="store_true")
parser.add_argument("--backup-dir", type=Path, help="Store database snapshots outside the project drive")
args = parser.parse_args()
output = root / "data/visionmatch-deployment"
output.mkdir(parents=True, exist_ok=True)
with sqlite3.connect((root / "data/matcher.db").as_uri() + "?mode=ro", uri=True) as source:
    active = source.execute("select id,status from batches where status in ('QUEUED','MATCHING')").fetchall()
    if active:
        raise SystemExit(f"Active batches: {active}. Do not restart.")
    rows = source.execute("select id,batch_id,category,current_path,matched_main_id,manual_product_id,review_status,status,match_method from matched_files order by id").fetchall()
    snapshot = {"files": len(rows), "mains": source.execute("select count(*) from main_images").fetchone()[0],
                "batches": source.execute("select count(*) from batches").fetchone()[0],
                "associations_sha256": hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()}
    if args.before:
        backup_root = args.backup_dir or output
        backup_root.mkdir(parents=True, exist_ok=True)
        backup = backup_root / ("matcher-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".db")
        with sqlite3.connect(backup) as destination:
            source.backup(destination)
        snapshot["backup"] = str(backup)
        (output / "before.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        before = json.loads((output / "before.json").read_text(encoding="utf-8"))
        snapshot["unchanged"] = all(snapshot[key] == before[key] for key in snapshot)
        (output / "after.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        assert snapshot["unchanged"], "Inventory changed; inspect changes, never auto-restore"
print(json.dumps(snapshot, ensure_ascii=True))
