import json
import httpx
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models import Batch, MainImage, MatchedFile
from app.services import visionmatch as vm


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "BASE_DIR", tmp_path)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as db:
        db.add(Batch(id=1, root_path=str(tmp_path), status="SCANNED", match_engine="VISIONMATCH",
                     match_config_snapshot=json.dumps({"visionmatch": {"max_calls": 1, "confirmed": True}})))
        for index, color in enumerate(("red", "blue"), 1):
            main = tmp_path / f"M{index}.png"
            source = tmp_path / f"query{index}.png"
            Image.new("RGB", (32, 32), color).save(main)
            Image.new("RGB", (32, 32), color).save(source)
            db.add(MainImage(id=index, batch_id=1, basename=f"M{index}", filename=main.name, file_path=str(main)))
            db.add(MatchedFile(id=index, batch_id=1, category="details", original_filename=source.name,
                               current_filename=source.name, current_path=str(source), original_path=str(source), extension=".png"))
        db.commit()
        yield db


def state_for(db, root):
    mains = list(db.scalars(select(MainImage)))
    files = list(db.scalars(select(MatchedFile)))
    mapping = {**vm.stage(mains, root / "main", "main"), **vm.stage(files, root / "target", "file")}
    images = []
    for index, (path, record) in enumerate(mapping.items(), 10):
        images.append({"id": index, "path": path, "sha256": record["hash"],
                       "local_id": ("A" if "main_" in Path(path).name else "B") + str(index)})
    state = {"images": images, "matches": [{"target_id": 12, "assigned_source_id": 10,
             "status": "MATCHED", "suggestion": {"decision": "MATCH", "evidence": ["same print"]},
             "candidates": [{"source_id": 10, "decision": "MATCH", "evidence": ["same print"]}]},
            {"target_id": 13, "assigned_source_id": None, "status": "UNMATCHED",
             "suggestion": {"decision": "UNCERTAIN"}, "candidates": []}]}
    return mapping, state


def test_writeback_fills_unknown_but_does_not_invent_model_scores(fixture, tmp_path):
    db = fixture
    mapping, state = state_for(db, tmp_path / "stage")
    assert vm.apply_result(db, 1, state, mapping, {"remote_batch_id": "abc", "run_id": "run"}) == 2
    a, b = list(db.scalars(select(MatchedFile).order_by(MatchedFile.id)))
    assert a.matched_main_id == 1 and a.match_method == "VISIONMATCH"
    assert a.top1_score is None and a.algorithm_score is None and not a.candidates
    assert b.matched_main_id == 2 and b.status == "AMBIGUOUS"
    detail = json.loads(b.match_details_json)
    assert detail["auto_filled"] is True and detail["model_assigned_main_id"] is None
    assert detail["suggestion"]["decision"] == "UNCERTAIN"
    assert b.algorithm_score is None and b.review_status == "PENDING"
    assert json.loads(a.match_details_json)["matrix_complete"] is False


def test_manual_and_exact_rows_reserve_columns(fixture):
    db = fixture
    a = db.get(MatchedFile, 1)
    a.match_method, a.matched_main_id = "EXACT_FILENAME", 1
    db.commit()
    assert vm.plan(db, 1)["folders"] == [{"category": "details", "mains": 1, "files": 1}]
    a.match_method, a.review_status = "LIGHTGLUE_COLOR", "CONFIRMED"
    assert vm.plan(db, 1)["folders"][0]["files"] == 1


def test_hash_mismatch_rejects_all_results(fixture, tmp_path):
    mapping, state = state_for(fixture, tmp_path / "stage")
    state["images"][0]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="身份"):
        vm.apply_result(fixture, 1, state, mapping, {"remote_batch_id": "abc", "run_id": "run"})
    assert fixture.get(MatchedFile, 1).matched_main_id is None


def test_changed_source_is_not_written_back(fixture, tmp_path):
    mapping, state = state_for(fixture, tmp_path / "stage")
    Image.new("RGB", (32, 32), "green").save(fixture.get(MatchedFile, 2).current_path)
    with pytest.raises(ValueError, match="源图片已变化"):
        vm.validate_result(state, mapping)


def test_duplicate_assignment_rejected(fixture, tmp_path):
    mapping, state = state_for(fixture, tmp_path / "stage")
    state["matches"][1]["assigned_source_id"] = 10
    with pytest.raises(ValueError, match="重复归属"):
        vm.apply_result(fixture, 1, state, mapping, {"remote_batch_id": "abc", "run_id": "run"})
    assert fixture.get(MatchedFile, 1).matched_main_id is None


def test_manual_correction_during_inference_survives(fixture, tmp_path):
    mapping, state = state_for(fixture, tmp_path / "stage")
    item = fixture.get(MatchedFile, 1)
    item.matched_main_id, item.review_status = 2, "CORRECTED"
    vm.apply_result(fixture, 1, state, mapping, {"remote_batch_id": "abc", "run_id": "run"})
    assert item.matched_main_id == 2 and item.review_status == "CORRECTED"


def test_budget_failure_does_not_call_backend(fixture, monkeypatch):
    fixture.get(Batch, 1).match_config_snapshot = "{}"
    monkeypatch.setattr(vm, "request", lambda *args: pytest.fail("must not call backend"))
    with pytest.raises(ValueError, match="预算不足"):
        vm.run_batch(fixture, 1)


def test_limit_is_explicit_not_candidate_truncation(fixture):
    for index in range(3, 52):
        fixture.add(MainImage(id=index, batch_id=1, basename=str(index), filename=f"{index}.png", file_path="unused"))
    fixture.flush()
    with pytest.raises(ValueError, match="不会自动截断"):
        vm.plan(fixture, 1)


def test_empty_selection_does_not_expand_to_all(fixture):
    assert vm.plan(fixture, 1, set())["calls_upper_bound"] == 0


def test_adapter_polls_and_writes_existing_backend_result(fixture, monkeypatch):
    db = fixture
    captured = {}
    def backend(method, path, body=None):
        if path == "/api/batches":
            return {"id": "backend123"}
        if path.endswith("/import"):
            captured.update(source=body["source"], target=body["targets"][0]["path"])
            return {}
        if path.endswith("/match"):
            assert body == {"confirmed_budget": True, "max_calls": 1, "concurrency": 4}
            return {"run_id": "run123"}
        images = []
        for role, folder in (("A", captured["source"]), ("B", captured["target"])):
            for file in sorted(Path(folder).iterdir()):
                images.append({"id": len(images) + 10, "path": str(file), "sha256": vm.file_hash(file), "local_id": role + str(len(images))})
        return {"runs": [{"id": "run123", "state": "COMPLETE"}], "jobs": [],
                "folders": [{"role": "TARGET", "state": "COMPLETE"}], "images": images,
                "matches": [{"target_id": 12, "assigned_source_id": 10, "status": "MATCHED", "suggestion": {"decision": "MATCH"}, "candidates": []},
                            {"target_id": 13, "assigned_source_id": 11, "status": "REVIEW", "suggestion": {"decision": "UNCERTAIN"}, "candidates": []}]}
    monkeypatch.setattr(vm, "request", backend)
    vm.run_batch(db, 1)
    assert db.get(Batch, 1).status == "COMPLETED"
    assert db.get(MatchedFile, 1).matched_main_id == 1
    assert db.get(MatchedFile, 2).status == "AMBIGUOUS"


def test_start_requires_budget_but_not_masks(fixture, monkeypatch):
    from fastapi import HTTPException
    import app.main as main
    monkeypatch.setattr(main.visionmatch, "status", lambda: {"id": "VISIONMATCH", "selectable": True})
    submitted = []
    monkeypatch.setattr(main.executor, "submit", lambda *args: submitted.append(args))
    with pytest.raises(HTTPException) as error:
        main.start_match(1, main.MatchStartRequest(match_engine="VISIONMATCH"), fixture)
    assert error.value.status_code == 409 and "预算" in error.value.detail
    assert submitted == []
    result = main.start_match(1, main.MatchStartRequest(match_engine="VISIONMATCH",
        visionmatch_confirmed_budget=True, visionmatch_max_calls=1), fixture)
    assert result["status"] == "QUEUED" and len(submitted) == 1
    assert result["folder_engines"]["details"]["engine"] == "VISIONMATCH"


def test_vacant_fill_uses_global_candidate_preferences(fixture, tmp_path):
    mapping, state = state_for(fixture, tmp_path / "stage")
    first, second = state["matches"]
    first.update(assigned_source_id=None, status="UNMATCHED", candidates=[
        {"source_id": 10, "decision": "UNCERTAIN"}, {"source_id": 11, "decision": "UNCERTAIN"}])
    second.update(candidates=[{"source_id": 10, "decision": "UNCERTAIN"}])
    vm.apply_result(fixture, 1, state, mapping, {"remote_batch_id": "abc", "run_id": "run"})
    assert fixture.get(MatchedFile, 1).matched_main_id == 2
    assert fixture.get(MatchedFile, 2).matched_main_id == 1
    assert all(json.loads(f.match_details_json)["auto_filled"] for f in fixture.scalars(select(MatchedFile)))


def test_fill_protects_existing_assignments_and_is_category_local():
    from types import SimpleNamespace
    def row(ident, category, main_id=None):
        return (SimpleNamespace(id=ident, category=category), main_id, {}, [])
    updates = [row(3, "scene"), row(1, "details", 1), row(2, "details")]
    assert vm.fill_vacant_slots(updates, {1: None, 2: None}, {("scene", 2)}) == {3: 1, 2: 2}


def test_fill_does_not_reuse_columns_when_sources_outnumber_mains():
    from types import SimpleNamespace
    rows = [(SimpleNamespace(id=i, category="details"), None, {}, []) for i in (3, 2, 1)]
    filled = vm.fill_vacant_slots(rows, {1: None, 2: None}, set())
    assert len(filled) == 2 and len(set(filled.values())) == 2
    assert filled == vm.fill_vacant_slots(list(reversed(rows)), {2: None, 1: None}, set())


def test_unknown_can_fill_a_main_outside_model_candidate_list():
    from types import SimpleNamespace
    row = (SimpleNamespace(id=1, category="details"), None, {},
           [{"main_image_id": 1, "decision": "UNCERTAIN"}])
    assert vm.fill_vacant_slots([row], {1: None, 2: None}, {("details", 1)}) == {1: 2}


def add_duplicate(db, tmp_path, kind="file"):
    import shutil
    if kind == "main":
        first = db.get(MainImage, 1)
        path = tmp_path / "duplicate_main.png"
        shutil.copyfile(first.file_path, path)
        obj = MainImage(id=3, batch_id=1, basename="M3", filename=path.name, file_path=str(path))
    else:
        first = db.get(MatchedFile, 1)
        path = tmp_path / "duplicate_query.png"
        shutil.copyfile(first.current_path, path)
        obj = MatchedFile(id=3, batch_id=1, category=first.category, original_filename=path.name,
                          current_filename=path.name, original_path=str(path), current_path=str(path), extension=".png")
    db.add(obj)
    db.commit()
    return obj


def test_duplicate_targets_use_one_staged_copy_and_keep_both_originals(fixture, tmp_path):
    extra = add_duplicate(fixture, tmp_path)
    mapping, state = state_for(fixture, tmp_path / "stage")
    assert len(mapping) == 4
    assert len(list((tmp_path / "stage" / "target").iterdir())) == 2
    assert Path(extra.current_path).is_file()
    assert next(row for row in mapping.values() if row.get("duplicates"))["duplicates"][0]["id"] == 3
    assert vm.validate_result(state, mapping)


def test_duplicate_alias_change_rejects_entire_writeback(fixture, tmp_path):
    extra = add_duplicate(fixture, tmp_path)
    mapping, state = state_for(fixture, tmp_path / "stage")
    Image.new("RGB", (32, 32), "green").save(extra.current_path)
    with pytest.raises(ValueError, match="源图片已变化"):
        vm.apply_result(fixture, 1, state, mapping, {})
    assert all(row.matched_main_id is None for row in fixture.scalars(select(MatchedFile)))


def test_duplicate_target_results_expand_and_require_review(fixture, tmp_path):
    add_duplicate(fixture, tmp_path)
    batch = fixture.get(Batch, 1)
    batch.match_config_snapshot = json.dumps({"mode": "FOLDER_SHARDED"})
    mapping, state = state_for(fixture, tmp_path / "stage")
    assert vm.apply_result(fixture, 1, state, mapping, {"remote_batch_id": "b", "run_id": "r"}) == 3
    for ident in (1, 3):
        row = fixture.get(MatchedFile, ident)
        assert row.matched_main_id == (1 if ident == 1 else None)
        assert row.status == ("AMBIGUOUS" if ident == 1 else "UNMATCHED")
        details = json.loads(row.match_details_json)
        assert details["duplicate_target_ids"] == [1, 3]
        assert details["duplicate_note"] and not details["auto_filled"]
    assert fixture.get(MatchedFile, 2).matched_main_id is None


def test_legacy_duplicate_target_does_not_take_unrelated_vacant_main(fixture, tmp_path):
    add_duplicate(fixture, tmp_path)
    mapping, state = state_for(fixture, tmp_path / "stage")
    vm.apply_result(fixture, 1, state, mapping, {"remote_batch_id": "b", "run_id": "r"})
    assert fixture.get(MatchedFile, 1).matched_main_id == 1
    assert fixture.get(MatchedFile, 3).matched_main_id is None
    assert fixture.get(MatchedFile, 2).matched_main_id == 2


def test_identical_mains_keep_all_candidates_without_guessing_product(fixture, tmp_path):
    add_duplicate(fixture, tmp_path, "main")
    mapping, state = state_for(fixture, tmp_path / "stage")
    vm.apply_result(fixture, 1, state, mapping, {"remote_batch_id": "b", "run_id": "r"})
    row = fixture.get(MatchedFile, 1)
    assert row.matched_main_id is None and row.status == "AMBIGUOUS"
    details = json.loads(row.match_details_json)
    assert details["duplicate_main_ids"] == [1, 3]
    assert {r["main_image_id"] for r in details["candidates"]} == {1, 3}


def test_folder_mode_uses_unique_backend_assignment_and_keeps_review_flag(fixture, tmp_path):
    fixture.get(Batch, 1).match_config_snapshot = json.dumps({"mode": "FOLDER_SHARDED"})
    mapping, state = state_for(fixture, tmp_path / "stage")
    state["matches"][0]["proposed_source_id"] = 10
    state["matches"][1].update(assigned_source_id=11, proposed_source_id=10,
        status="REVIEW", suggestion={"decision": "MATCH", "evidence": ["same design"], "needs_review": False})
    vm.apply_result(fixture, 1, state, mapping, {"remote_batch_id": "b", "run_id": "r"})
    assert [fixture.get(MatchedFile, i).matched_main_id for i in (1, 2)] == [1, 2]
    assert fixture.get(MatchedFile, 2).status == "AMBIGUOUS"


def test_folder_mode_reserves_locked_mains_in_same_category(fixture):
    fixture.get(Batch, 1).match_config_snapshot = json.dumps({"mode": "FOLDER_SHARDED"})
    locked = fixture.get(MatchedFile, 1)
    locked.match_method, locked.matched_main_id = "EXACT_FILENAME", 1
    assert vm.plan(fixture, 1)["folders"] == [{"category": "details", "mains": 1, "files": 1}]


def test_uncertain_proposal_does_not_occupy_a_main_already_assigned_in_same_folder(fixture, tmp_path):
    fixture.get(Batch, 1).match_config_snapshot = json.dumps({"mode": "FOLDER_SHARDED"})
    mapping, state = state_for(fixture, tmp_path / "stage")
    state["matches"][1].update(proposed_source_id=10, assigned_source_id=None,
        suggestion={"decision": "UNCERTAIN", "needs_review": True})
    vm.apply_result(fixture, 1, state, mapping, {"remote_batch_id": "b", "run_id": "r"})
    assert fixture.get(MatchedFile, 1).matched_main_id == 1
    assert fixture.get(MatchedFile, 2).matched_main_id is None
    assert not json.loads(fixture.get(MatchedFile, 2).match_details_json)["auto_filled"]


def test_marked_conflict_files_are_not_rematched(fixture):
    fixture.get(MatchedFile, 1).status = "CONFLICT_MARKED"
    assert vm.plan(fixture, 1)["folders"] == [{"category": "details", "mains": 2, "files": 1}]


def test_folder_configuration_forbids_multiple_files_per_main(fixture, monkeypatch):
    import app.main as main
    monkeypatch.setattr(main.qwen_lab, "status", lambda: {"reachable": False})
    snapshot = main._configure_folder_engines(fixture.get(Batch, 1), fixture, {"details": {"engine": "VISIONMATCH"}})
    assert snapshot["duplicate_within_folder"] is False


def test_repeated_remote_path_does_not_pass_coverage_check(fixture, tmp_path):
    mapping, state = state_for(fixture, tmp_path / "stage")
    state["images"][1]["path"] = state["images"][0]["path"]
    state["images"][1]["sha256"] = state["images"][0]["sha256"]
    with pytest.raises(ValueError, match="图片重复"):
        vm.validate_result(state, mapping)


def test_status_timeout_retries_read_only_without_repeating_model_submission(monkeypatch):
    calls = []
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def request(self, method, url, json=None):
            calls.append(method)
            if len(calls) == 1:
                raise httpx.ReadTimeout("busy")
            return httpx.Response(200, json={"ok": True}, request=httpx.Request(method, url))
    monkeypatch.setattr(vm.httpx, "Client", Client)
    monkeypatch.setattr(vm.time, "sleep", lambda _: None)
    assert vm.request("GET", "/api/batches/x") == {"ok": True}
    assert calls == ["GET", "GET"]
    calls.clear()
    with pytest.raises(ValueError, match="连接或响应异常"):
        vm.request("POST", "/api/batches/x/match", {})
    assert calls == ["POST"]


def test_completed_folder_survives_later_folder_failure(fixture, monkeypatch):
    fixture.get(MatchedFile, 2).category = "scene"
    fixture.get(Batch, 1).match_config_snapshot = json.dumps({"mode": "FOLDER_SHARDED", "visionmatch": {"max_calls": 2}})
    fixture.commit()
    captured = {}
    created = []
    def backend(method, path, body=None):
        if path == "/api/batches":
            created.append(body)
            if len(created) == 2:
                raise ValueError("later folder unavailable")
            return {"id": "first"}
        if path.endswith("/import"):
            captured.update(source=body["source"], target=body["targets"][0]["path"])
            return {}
        if path.endswith("/match"):
            return {"run_id": "first-run"}
        images = []
        for role, folder in (("A", captured["source"]), ("B", captured["target"])):
            for file in sorted(Path(folder).iterdir()):
                images.append({"id": len(images) + 10, "path": str(file), "sha256": vm.file_hash(file), "local_id": role + str(len(images))})
        return {"runs": [{"id": "first-run", "state": "COMPLETE"}], "jobs": [],
                "folders": [{"role": "TARGET", "state": "COMPLETE"}], "images": images,
                "matches": [{"target_id": 12, "assigned_source_id": 10, "status": "MATCHED",
                             "suggestion": {"decision": "MATCH", "evidence": ["same print"]}, "candidates": []}]}
    monkeypatch.setattr(vm, "request", backend)
    with pytest.raises(ValueError, match="later folder"):
        vm.run_batch(fixture, 1)
    assert fixture.get(Batch, 1).status == "FAILED"
    assert fixture.get(MatchedFile, 1).match_method == "VISIONMATCH"
    assert fixture.get(MatchedFile, 2).match_method is None


def parallel_folders(db, tmp_path, count=6):
    for index in range(1, count + 1):
        if index <= 2:
            db.get(MatchedFile, index).category = f"folder{index}"
        else:
            path = tmp_path / f"query{index}.png"
            Image.new("RGB", (32, 32), (index * 23, index * 11, 80)).save(path)
            db.add(MatchedFile(id=index, batch_id=1, category=f"folder{index}", original_filename=path.name,
                current_filename=path.name, original_path=str(path), current_path=str(path), extension=".png"))
    db.get(Batch, 1).match_config_snapshot = json.dumps({"mode": "FOLDER_SHARDED", "visionmatch": {"max_calls": count}})
    db.commit()


def parallel_backend(monkeypatch, *, unknown_submission=None, failed_run=None, rolling=False):
    records = {}
    active = set()
    observed = {"peak": 0, "submissions": [], "first_poll_submitted": None, "active_at_fifth": None}
    fourth_done = False

    def backend(method, path, body=None):
        nonlocal fourth_done
        if path == "/api/batches":
            ident = str(len(records) + 1)
            records[ident] = {}
            return {"id": ident}
        ident = path.split("/")[3]
        record = records[ident]
        if path.endswith("/import"):
            record.update(source=body["source"], target=body["targets"][0]["path"])
            return {}
        if path.endswith("/match"):
            assert body == {"confirmed_budget": True, "max_calls": 1, "concurrency": 4}
            observed["submissions"].append(ident)
            if ident == unknown_submission:
                raise ValueError("VisionMatch 8066 连接或响应异常：timed out")
            if ident == "5":
                observed["active_at_fifth"] = set(active)
            active.add(ident)
            observed["peak"] = max(observed["peak"], len(active))
            return {"run_id": f"run{ident}"}
        if observed["first_poll_submitted"] is None:
            observed["first_poll_submitted"] = len(observed["submissions"])
        if rolling and not fourth_done and ident in {"1", "2", "3"}:
            return {"runs": [{"id": f"run{ident}", "state": "RUNNING"}], "jobs": []}
        if ident == "4":
            fourth_done = True
        active.discard(ident)
        if ident == failed_run:
            return {"runs": [{"id": f"run{ident}", "state": "COMPLETE_WITH_ERRORS"}],
                    "jobs": [{"run_id": f"run{ident}", "state": "FAILED", "error": "model timeout"}]}
        images = []
        for role, folder in (("A", record["source"]), ("B", record["target"])):
            for file in sorted(Path(folder).iterdir()):
                images.append({"id": len(images) + 10, "path": str(file), "sha256": vm.file_hash(file),
                               "local_id": role + str(len(images))})
        return {"runs": [{"id": f"run{ident}", "state": "COMPLETE"}], "jobs": [],
                "folders": [{"role": "TARGET", "state": "COMPLETE"}], "images": images,
                "matches": [{"target_id": 12, "assigned_source_id": 10, "status": "MATCHED",
                             "suggestion": {"decision": "MATCH", "evidence": ["fixture only"]}, "candidates": []}]}
    monkeypatch.setattr(vm, "request", backend)
    monkeypatch.setattr(vm.time, "sleep", lambda _: None)
    return records, observed


def test_four_in_flight_and_refill_without_waiting_for_the_other_three(fixture, tmp_path, monkeypatch):
    parallel_folders(fixture, tmp_path)
    _, observed = parallel_backend(monkeypatch, rolling=True)
    vm.run_batch(fixture, 1)
    assert observed["peak"] == 4
    assert observed["first_poll_submitted"] == 4
    assert observed["active_at_fifth"] == {"1", "2", "3"}
    assert len(observed["submissions"]) == 6
    assert fixture.get(Batch, 1).status == "COMPLETED"
    assert all(f.match_method == "VISIONMATCH" for f in fixture.scalars(select(MatchedFile)))
    manifest = vm.get_link(1)
    assert manifest["concurrency"] == 4 and not manifest["in_progress"]
    assert len(manifest["runs"]) == 6 and all(r["state"] == "COMPLETE" for r in manifest["runs"])


def test_unknown_post_is_not_repeated_and_other_inflight_results_survive(fixture, tmp_path, monkeypatch):
    parallel_folders(fixture, tmp_path)
    _, observed = parallel_backend(monkeypatch, unknown_submission="4")
    with pytest.raises(ValueError, match="连接或响应异常"):
        vm.run_batch(fixture, 1)
    assert observed["submissions"] == ["1", "2", "3", "4"]
    assert sum(f.match_method == "VISIONMATCH" for f in fixture.scalars(select(MatchedFile))) == 3
    assert vm.get_link(1)["runs"][-1]["state"] == "SUBMISSION_UNKNOWN"


def test_one_model_failure_keeps_other_folders_within_original_budget(fixture, tmp_path, monkeypatch):
    parallel_folders(fixture, tmp_path)
    _, observed = parallel_backend(monkeypatch, failed_run="2")
    with pytest.raises(ValueError, match="model timeout"):
        vm.run_batch(fixture, 1)
    assert len(observed["submissions"]) == 6 and observed["peak"] == 4
    assert sum(f.match_method == "VISIONMATCH" for f in fixture.scalars(select(MatchedFile))) == 5
    assert fixture.get(Batch, 1).status == "FAILED"


@pytest.mark.parametrize("action", ["pause", "resume", "cancel"])
def test_parallel_controls_reach_all_four_active_runs(fixture, monkeypatch, action):
    links = [{"category": str(i), "remote_batch_id": str(i), "run_id": f"run{i}"} for i in range(5)]
    vm.save_link(1, {"in_progress": True, "runs": links})
    calls = []
    def backend(method, path, body=None):
        if method == "GET":
            ident = path.rsplit("/", 1)[1]
            return {"runs": [{"id": f"run{ident}", "state": "COMPLETE" if ident == "4" else "PAUSED" if action == "resume" else "RUNNING",
                              "used_calls": 0, "max_calls": 1}]}
        calls.append(path)
        return {"ok": True}
    monkeypatch.setattr(vm, "request", backend)
    assert vm.control(1, action)["controlled_runs"] == 4
    assert set(calls) == {f"/api/runs/run{i}/{action}" for i in range(4)}
    assert vm.control_state(1) == {"paused": action == "pause", "cancelled": action == "cancel"}


def test_prepare_phase_can_be_paused_before_any_model_call(fixture, monkeypatch):
    vm.save_link(1, {"in_progress": True, "runs": []})
    monkeypatch.setattr(vm, "request", lambda *args: pytest.fail("no backend run exists yet"))
    assert vm.control(1, "pause")["controlled_runs"] == 0
    assert vm.control_state(1)["paused"]
