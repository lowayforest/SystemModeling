"""Generate the 4-stage M-sequence used as input in textbook Example 5.3.

The register and feedback follow textbook Example 2.6 (printed page 55):
initial state C1..C4 = 1010, feedback = C3 XOR C4, output = C4.
No random seed is needed for this deterministic sequence.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INITIAL_STATE = (1, 0, 1, 0)
PERIOD = 2**4 - 1


def generate(length: int) -> list[dict[str, int]]:
    if length < 1:
        raise ValueError("length must be positive")

    state = list(INITIAL_STATE)
    rows: list[dict[str, int]] = []
    for k in range(1, length + 1):
        c1, c2, c3, c4 = state
        feedback = c3 ^ c4
        bit = c4
        rows.append(
            {
                "k": k,
                "C1": c1,
                "C2": c2,
                "C3": c3,
                "C4": c4,
                "feedback": feedback,
                "bit": bit,
                "u": 2 * bit - 1,
            }
        )
        state = [feedback, c1, c2, c3]
    return rows


def check_one_period(rows: list[dict[str, int]]) -> None:
    first = rows[:PERIOD]
    states = [tuple(row[key] for key in ("C1", "C2", "C3", "C4")) for row in first]
    assert len(states) == PERIOD and len(set(states)) == PERIOD
    assert (first[0]["C1"], first[0]["C2"], first[0]["C3"], first[0]["C4"]) == INITIAL_STATE
    assert tuple(generate(PERIOD + 1)[-1][key] for key in ("C1", "C2", "C3", "C4")) == INITIAL_STATE

    u = [row["u"] for row in first]
    correlations = [
        sum(u[k] * u[(k + lag) % PERIOD] for k in range(PERIOD)) / PERIOD
        for lag in range(PERIOD)
    ]
    assert correlations[0] == 1
    assert all(abs(value + 1 / PERIOD) < 1e-12 for value in correlations[1:])


def write_csv(rows: list[dict[str, int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=400, help="number of input samples")
    parser.add_argument("--output", type=Path, default=ROOT / "output/m_sequence_400.csv")
    args = parser.parse_args()

    rows = generate(args.length)
    check_one_period(generate(PERIOD))
    write_csv(rows, args.output)
    bits = "".join(str(row["bit"]) for row in rows[:PERIOD])
    print(f"First period (0/1): {bits}")
    print(f"Period: {PERIOD}; first-period mean of +/-1 input: {sum(row['u'] for row in rows[:PERIOD]) / PERIOD:.6f}")
    print(f"Wrote {len(rows)} input samples to {args.output}")


if __name__ == "__main__":
    main()
