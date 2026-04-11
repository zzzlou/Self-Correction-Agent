from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize_serialized_episodes(episodes: list[dict]) -> dict[str, float]:
    total = len(episodes)
    if total == 0:
        return {
            "num_examples": 0,
            "first_pass_accuracy": 0.0,
            "final_accuracy": 0.0,
            "correction_rate": 0.0,
            "second_attempt_rate": 0.0,
            "pass_at_1": 0.0,
            "pass_at_2": 0.0,
        }

    first_try_correct = 0
    final_correct = 0
    used_second_attempt = 0
    corrected_on_second_attempt = 0
    first_try_failures = 0

    for episode in episodes:
        attempt_1_correct = bool(episode.get("attempt_1_correct", False))
        final_is_correct = bool(episode.get("final_correct", False))
        corrected = bool(episode.get("corrected_on_second_attempt", False))
        used_retry = episode.get("attempt_2_response") is not None

        first_try_correct += int(attempt_1_correct)
        final_correct += int(final_is_correct)
        used_second_attempt += int(used_retry)
        corrected_on_second_attempt += int(corrected)
        first_try_failures += int(not attempt_1_correct)

    correction_rate = corrected_on_second_attempt / first_try_failures if first_try_failures else 0.0
    first_pass_accuracy = first_try_correct / total
    final_accuracy = final_correct / total
    second_attempt_rate = used_second_attempt / total
    return {
        "num_examples": total,
        "first_pass_accuracy": first_pass_accuracy,
        "final_accuracy": final_accuracy,
        "correction_rate": correction_rate,
        "second_attempt_rate": second_attempt_rate,
        "pass_at_1": first_pass_accuracy,
        "pass_at_2": final_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute offline two-attempt math metrics from episodes.json.")
    parser.add_argument(
        "episodes_path",
        nargs="?",
        default="logs/two_attempt_math/episodes.json",
        help="Path to the serialized episodes JSON produced by projects.two_attempt_math.eval",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the computed summary JSON",
    )
    args = parser.parse_args()

    episodes_path = Path(args.episodes_path)
    with episodes_path.open("r", encoding="utf-8") as f:
        episodes = json.load(f)

    summary = summarize_serialized_episodes(episodes)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
