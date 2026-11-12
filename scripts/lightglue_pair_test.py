from __future__ import annotations

import argparse
import json

from app.services.lightglue_matching import pair_test


def main() -> None:
    parser = argparse.ArgumentParser(description="ALIKED + LightGlue 正确/错误主图 Pair Test")
    parser.add_argument("query")
    parser.add_argument("correct_main")
    parser.add_argument("wrong_main")
    args = parser.parse_args()
    print(json.dumps({
        "correct": pair_test(args.query, args.correct_main),
        "wrong": pair_test(args.query, args.wrong_main),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
