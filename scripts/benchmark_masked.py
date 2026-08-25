from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app.models import DerivedImageAsset, MainImage, MatchCandidate, MatchedFile
from app.services.matcher import match_batch
from app.services.psd import PHashProvider


def _path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _number(filename: str) -> int:
    match = re.search(r"\d+", Path(filename).stem)
    return int(match.group()) if match else 10**9


def _source_snapshot(files: list[MatchedFile]) -> dict[str, tuple[int, int]]:
    snapshot = {}
    for item in files:
        path = Path(item.current_path)
        if path.is_file():
            stat = path.stat()
            snapshot[_path_key(str(path))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _reference_truth(
    db,
    reference_batch: int,
    target_mains: list[MainImage],
) -> dict[str, str]:
    """Recover the known numeric delivery order from the untouched first scan."""
    reference_files = list(db.scalars(select(MatchedFile).where(
        MatchedFile.batch_id == reference_batch,
        MatchedFile.category != "PSD",
    )))
    by_category: dict[str, list[MatchedFile]] = defaultdict(list)
    for item in reference_files:
        by_category[item.category].append(item)
    ordered_products = [main.basename for main in sorted(target_mains, key=lambda row: row.id)]
    truth = {}
    for items in by_category.values():
        for item, product in zip(
            sorted(items, key=lambda row: (_number(row.original_filename), row.id)),
            ordered_products,
            strict=False,
        ):
            truth[_path_key(item.current_path)] = product
    return truth


def _device() -> dict:
    try:
        import torch

        return {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
    except Exception as exc:
        return {"cuda_available": False, "device": "unknown", "error": str(exc)}


def _evaluate(
    db,
    files: list[MatchedFile],
    mains: list[MainImage],
    truth: dict[str, str],
    old_psd: dict[int, int | None],
) -> dict:
    main_names = {main.id: main.basename for main in mains}
    candidates: dict[int, list] = defaultdict(list)
    file_ids = [item.id for item in files]
    if file_ids:
        rows = db.scalars(select(MatchCandidate).where(
            MatchCandidate.matched_file_id.in_(file_ids)
        ).order_by(MatchCandidate.matched_file_id, MatchCandidate.rank)).all()
        for row in rows:
            candidates[row.matched_file_id].append(row)

    categories: dict[str, dict] = {}
    grouped: dict[str, list[MatchedFile]] = defaultdict(list)
    for item in files:
        grouped[item.category].append(item)
    overall = Counter()
    changed = Counter()
    spectator_report = None
    for category, items in sorted(grouped.items()):
        statuses = Counter(item.status for item in items)
        assignments = Counter(
            item.algorithm_main_id for item in items if item.algorithm_main_id is not None
        )
        methods = Counter(item.match_method or "unknown" for item in items)
        row = {
            "source_files": len(items),
            "statuses": dict(statuses),
            "unique_algorithm_products": len(assignments),
            "duplicate_algorithm_assignments": sum(value - 1 for value in assignments.values()),
            "match_methods": dict(methods),
        }
        if category == "PSD":
            comparable = [item for item in items if old_psd.get(item.id) is not None]
            same = sum(item.algorithm_main_id == old_psd[item.id] for item in comparable)
            distances = [item.phash_distance for item in items if item.phash_distance is not None]
            phash = PHashProvider()
            main_hashes = [phash.compute_path(Path(main.file_path)) for main in mains]
            pure_phash = {}
            for item in items:
                preview = db.scalar(select(DerivedImageAsset).where(
                    DerivedImageAsset.parent_file_id == item.id,
                    DerivedImageAsset.asset_type == "PSD_PREVIEW",
                ).order_by(DerivedImageAsset.id.desc()))
                query_hash = phash.compute_path(
                    Path(preview.asset_path) if preview else Path(item.current_path)
                )
                pure_phash[item.id] = mains[
                    min(
                        range(len(mains)),
                        key=lambda index: phash.distance(query_hash, main_hashes[index]),
                    )
                ].id
            pure_correct = sum(
                pure_phash.get(item.id) == old_psd.get(item.id) for item in comparable
            )
            row.update({
                "previous_assignment_comparable": len(comparable),
                "pure_phash_top1_correct": pure_correct,
                "embedding_assisted": sum(
                    item.match_method == "PHASH_TIE_BREAK_EMBEDDING" for item in items
                ),
                "same_as_previous_assignment": same,
                "same_as_previous_rate": same / len(comparable) if comparable else None,
                "phash_distance_min": min(distances) if distances else None,
                "phash_distance_max": max(distances) if distances else None,
            })
        else:
            known = [item for item in items if _path_key(item.current_path) in truth]
            baseline_correct = 0
            masked_top1_correct = 0
            assigned_correct = 0
            baseline_products = Counter()
            masked_products = Counter()
            for item in known:
                expected = truth[_path_key(item.current_path)]
                baseline_ok = main_names.get(item.baseline_top1_main_id) == expected
                top1 = candidates[item.id][0].main_image_id if candidates[item.id] else None
                baseline_products[item.baseline_top1_main_id] += 1
                masked_products[top1] += 1
                masked_ok = main_names.get(top1) == expected
                assigned_ok = main_names.get(item.algorithm_main_id) == expected
                baseline_correct += baseline_ok
                masked_top1_correct += masked_ok
                assigned_correct += assigned_ok
                changed[
                    "improved" if masked_ok and not baseline_ok
                    else "regressed" if baseline_ok and not masked_ok
                    else "unchanged_correct" if masked_ok
                    else "unchanged_wrong"
                ] += 1
            row.update({
                "ground_truth_files": len(known),
                "full_openclip_top1_correct": baseline_correct,
                "full_openclip_top1_accuracy": baseline_correct / len(known) if known else None,
                "full_openclip_unique_products": len(baseline_products),
                "full_openclip_duplicate_top1": sum(
                    value - 1 for value in baseline_products.values()
                ),
                "masked_openclip_top1_correct": masked_top1_correct,
                "masked_openclip_top1_accuracy": masked_top1_correct / len(known) if known else None,
                "masked_openclip_unique_products": len(masked_products),
                "masked_openclip_duplicate_top1": sum(
                    value - 1 for value in masked_products.values()
                ),
                "hungarian_assigned_correct": assigned_correct,
                "hungarian_assigned_accuracy": assigned_correct / len(known) if known else None,
            })
            overall["ground_truth_files"] += len(known)
            overall["full_openclip_top1_correct"] += baseline_correct
            overall["masked_openclip_top1_correct"] += masked_top1_correct
            overall["hungarian_assigned_correct"] += assigned_correct
            if category == "观赛图":
                arizona = next(
                    (main for main in mains if main.basename.strip() == "Arizona"), None
                )
                focus = next(
                    (item for item in items if item.original_filename == "37.jpg"), None
                )
                focus_candidates = candidates.get(focus.id, []) if focus else []
                spectator_report = {
                    "source_files": len(items),
                    "full_openclip_unique_products": len(baseline_products),
                    "masked_openclip_unique_products": len(masked_products),
                    "hungarian_unique_products": len(assignments),
                    "arizona_full_openclip_top1_count": sum(
                        item.baseline_top1_main_id == arizona.id for item in items
                    ) if arizona else None,
                    "arizona_masked_top1_count": sum(
                        candidates[item.id][0].main_image_id == arizona.id
                        for item in items if candidates[item.id]
                    ) if arizona else None,
                    "arizona_hungarian_count": sum(
                        item.algorithm_main_id == arizona.id for item in items
                    ) if arizona else None,
                    "buffalo_37": {
                        "file_id": focus.id,
                        "full_openclip_top1": main_names.get(focus.baseline_top1_main_id),
                        "full_openclip_top1_score": focus.baseline_top1_score,
                        "full_openclip_top2_score": focus.baseline_top2_score,
                        "full_openclip_top1_top2_margin": (
                            focus.baseline_top1_score - focus.baseline_top2_score
                            if focus.baseline_top1_score is not None
                            and focus.baseline_top2_score is not None else None
                        ),
                        "masked_top1": (
                            main_names.get(focus_candidates[0].main_image_id)
                            if focus_candidates else None
                        ),
                        "masked_top1_score": (
                            focus_candidates[0].similarity if focus_candidates else None
                        ),
                        "masked_top2": (
                            main_names.get(focus_candidates[1].main_image_id)
                            if len(focus_candidates) > 1 else None
                        ),
                        "masked_top2_score": (
                            focus_candidates[1].similarity
                            if len(focus_candidates) > 1 else None
                        ),
                        "assigned_score": focus.algorithm_score,
                        "row_margin": focus.row_margin,
                        "column_margin": focus.column_margin,
                        "final_product": main_names.get(focus.algorithm_main_id),
                        "status": focus.status,
                    } if focus else None,
                }
        categories[category] = row
    denominator = overall["ground_truth_files"]
    return {
        "categories": categories,
        "non_psd_overall": {
            **dict(overall),
            "full_openclip_top1_accuracy": (
                overall["full_openclip_top1_correct"] / denominator if denominator else None
            ),
            "masked_openclip_top1_accuracy": (
                overall["masked_openclip_top1_correct"] / denominator if denominator else None
            ),
            "hungarian_assigned_accuracy": (
                overall["hungarian_assigned_correct"] / denominator if denominator else None
            ),
            "top1_change_vs_baseline": dict(changed),
        },
        "spectator_focus": spectator_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--reference-batch", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    init_db()
    with SessionLocal() as db:
        mains = list(db.scalars(select(MainImage).where(
            MainImage.batch_id == args.batch
        ).order_by(MainImage.id)))
        files = list(db.scalars(select(MatchedFile).where(
            MatchedFile.batch_id == args.batch
        ).order_by(MatchedFile.id)))
        truth = _reference_truth(db, args.reference_batch, mains)
        old_psd = {
            item.id: item.algorithm_main_id for item in files if item.category == "PSD"
        }
        before = _source_snapshot(files)
        match_result = None if args.no_run else match_batch(db, args.batch)
        db.expire_all()
        files = list(db.scalars(select(MatchedFile).where(
            MatchedFile.batch_id == args.batch
        ).order_by(MatchedFile.id)))
        after = _source_snapshot(files)
        report = {
            "batch_id": args.batch,
            "reference_batch_id": args.reference_batch,
            "main_count": len(mains),
            "source_file_count": len(files),
            "device": _device(),
            "match_result": match_result,
            "safety": {
                "physical_files_before": len(before),
                "physical_files_after": len(after),
                "source_paths_unchanged": set(before) == set(after),
                "source_bytes_and_mtime_unchanged": before == after,
            },
            **_evaluate(db, files, mains, truth, old_psd),
        }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
