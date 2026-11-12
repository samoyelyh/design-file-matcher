"""Strict, case-sensitive filename-stem association before visual inference."""
import json
from collections import defaultdict
from pathlib import PureWindowsPath

from sqlalchemy import select

from app.models import MainImage, MatchedFile
from app.config import IMAGE_EXTENSIONS, is_excluded_file_category

EXACT = "EXACT_FILENAME"


def filename_stem(filename):
    return PureWindowsPath(filename).stem


def associate_filenames(db, batch_id):
    index = defaultdict(list)
    for main in db.scalars(select(MainImage).where(MainImage.batch_id == batch_id)):
        index[filename_stem(main.filename)].append(main)
    files = list(db.scalars(select(MatchedFile).where(MatchedFile.batch_id == batch_id)))
    occupied = defaultdict(list)
    for item in files:
        if item.matched_main_id:
            occupied[(item.category, item.matched_main_id)].append(item)
    count = 0
    for item in files:
        if is_excluded_file_category(item.category) or PureWindowsPath(item.original_filename).suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        # Re-scans must preserve manual decisions, renames and existing results.
        if item.match_method or item.matched_main_id or item.review_status in {"CONFIRMED", "CORRECTED"}:
            continue
        stem = filename_stem(item.original_filename)
        candidates = index.get(stem, [])
        if not candidates:
            continue
        details = {"source_filename": item.original_filename, "normalized_filename": stem}
        if len(candidates) != 1:
            item.match_method = "FILENAME_MATCH_AMBIGUOUS"
            item.algorithm_status = "FILENAME_MATCH_AMBIGUOUS"
            item.status = "AMBIGUOUS"
            item.is_conflict = True
            details["candidate_main_ids"] = [main.id for main in candidates]
        else:
            main = candidates[0]
            key = (item.category, main.id)
            if occupied[key]:
                item.match_method = "FILENAME_MATCH_AMBIGUOUS"
                item.algorithm_status = "FILENAME_MATCH_AMBIGUOUS"
                item.status = "AMBIGUOUS"
                item.is_conflict = True
                details["reason"] = "同分类已有图片关联此主图"
            else:
                item.matched_main_id = item.algorithm_main_id = main.id
                item.matched_main_basename = main.basename
                item.match_method = EXACT
                item.algorithm_status = "EXACT_FILENAME_MATCH"
                item.status = "MATCHED"
                item.is_low_confidence = item.is_conflict = False
                details.update(matched_main_image_filename=main.filename,
                               matched_main_image_path=main.file_path)
                occupied[key].append(item)
                count += 1
        item.match_details_json = json.dumps(details, ensure_ascii=False)
    db.flush()
    return count
