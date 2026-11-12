"""One explicit real-call integration check, using an isolated SQLite database."""
import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.db import Base
from app.models import Batch, MainImage, MatchedFile
from app.services import visionmatch as vm

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--execute-one-call", action="store_true")
args = parser.parse_args()
if not args.execute_one_call:
    raise SystemExit("Requires explicit --execute-one-call; no model was called")
root = vm.BASE_DIR / "data/visionmatch-smoke" / uuid.uuid4().hex
root.mkdir(parents=True)
vm.BASE_DIR = root
source = args.source.resolve()
before = vm.file_hash(source)
query = root / ("probe-query" + source.suffix)
shutil.copyfile(source, query)
engine = create_engine("sqlite:///" + (root / "smoke.db").as_posix())
Base.metadata.create_all(engine)
with Session(engine, expire_on_commit=False) as db:
    db.add(Batch(id=1, root_path=str(root), status="SCANNED", match_engine="VISIONMATCH",
                 match_config_snapshot=json.dumps({"visionmatch": {"max_calls": 1, "confirmed": True}})))
    db.add(MainImage(id=1, batch_id=1, basename=source.stem, filename=source.name, file_path=str(source)))
    db.add(MatchedFile(id=1, batch_id=1, category="integration-probe", original_filename=query.name,
                       current_filename=query.name, current_path=str(query), original_path=str(query), extension=query.suffix))
    db.commit()
    vm.run_batch(db, 1)
    item = db.get(MatchedFile, 1)
    link = vm.get_link(1)
    backend = vm.request("GET", f"/api/batches/{link['remote_batch_id']}")
    result = {"batch_status": db.get(Batch, 1).status, "matched_main_id": item.matched_main_id,
              "match_method": item.match_method, "status": item.status,
              "score": item.algorithm_score, "details": json.loads(item.match_details_json),
              "source_unchanged": before == vm.file_hash(source),
              "backend_runs": backend["runs"], "jobs": backend["jobs"],
              "scope": "one identical-image functional probe, not matching accuracy acceptance"}
    (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result_path": str(root / "result.json"), "status": item.status,
                      "matched_main_id": item.matched_main_id, "source_unchanged": result["source_unchanged"]}, ensure_ascii=True))
    assert item.matched_main_id == 1 and item.match_method == "VISIONMATCH"
    assert item.algorithm_score is None and result["source_unchanged"]
