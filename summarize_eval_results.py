#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


DEFAULT_SEEDS = (1, 2, 3, 4, 5)
BASE_DIR = Path(__file__).resolve().parent
METHOD_NAME_MAP = {
    "baseline": "baseline",
    "ens_post_tanh": "ens_post",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize eval_result CSV files into one per-env result table."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=BASE_DIR / "eval_result",
        help="Directory containing per-run eval CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "eval_result_summary.csv",
        help="Output summary CSV path.",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=3,
        help="Average the last N evaluation rows from each per-run CSV.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Seeds to include as columns.",
    )
    return parser.parse_args()


def parse_result_filename(path):
    match = re.match(r"(.+)_seed_(\d+)_(.+)\.csv$", path.name)
    if match is None:
        return None

    env_name, seed, raw_method = match.groups()
    method = METHOD_NAME_MAP.get(raw_method)
    if method is None:
        return None

    return env_name, int(seed), method


def mean_last_scores(path, last_n):
    rows = []
    with path.open(newline="") as fin:
        reader = csv.DictReader(fin)
        reader.fieldnames = [
            field.strip() if field is not None else field
            for field in (reader.fieldnames or [])
        ]
        if "episode_avg_return" not in reader.fieldnames:
            raise ValueError(f"{path} does not contain column 'episode_avg_return'")

        for row in reader:
            stripped_row = {key.strip(): value for key, value in row.items() if key is not None}
            score = stripped_row.get("episode_avg_return", "").strip()
            if score:
                rows.append(float(score))

    if not rows:
        return None

    selected = rows[-last_n:]
    return sum(selected) / len(selected)


def format_score(score):
    if score is None:
        return ""
    return f"{score:.6f}".rstrip("0").rstrip(".")


def collect_scores(result_dir, seeds, last_n):
    scores = {}
    for path in sorted(result_dir.glob("*.csv")):
        parsed = parse_result_filename(path)
        if parsed is None:
            continue

        env_name, seed, method = parsed
        if seed not in seeds:
            continue

        score = mean_last_scores(path, last_n)
        scores.setdefault(env_name, {}).setdefault(method, {})[seed] = score

    return scores


def write_summary(scores, seeds, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    methods = ("baseline", "ens_post")

    with output_path.open("w", newline="") as fout:
        writer = csv.writer(fout)
        for env_index, env_name in enumerate(sorted(scores)):
            if env_index > 0:
                writer.writerow([])

            writer.writerow([env_name, *seeds, "avg_score"])
            for method in methods:
                seed_scores = [scores[env_name].get(method, {}).get(seed) for seed in seeds]
                present_scores = [score for score in seed_scores if score is not None]
                avg_score = (
                    sum(present_scores) / len(present_scores)
                    if present_scores
                    else None
                )
                writer.writerow(
                    [method, *[format_score(score) for score in seed_scores], format_score(avg_score)]
                )


def main():
    args = parse_args()
    if args.last_n <= 0:
        raise ValueError("--last-n must be positive")
    if not args.result_dir.is_dir():
        raise FileNotFoundError(f"Result directory not found: {args.result_dir}")

    seeds = tuple(args.seeds)
    scores = collect_scores(args.result_dir, seeds, args.last_n)
    write_summary(scores, seeds, args.output)
    print(f"Wrote {args.output} with {len(scores)} environments.")


if __name__ == "__main__":
    main()
