import argparse
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

VARIABLES_REGEX = re.compile(r"v(\d+)")


def _count_variables(formula: str) -> int:
    matches = VARIABLES_REGEX.findall(formula)
    return max((int(i) for i in matches), default=0) + 1


class Sampler:
    def __init__(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed=seed)

    def gen_values(self, n_values: int) -> tuple[np.uint32, ...]:
        values = self.rng.integers(
            np.iinfo(np.uint32).max,
            size=n_values,
            dtype=np.uint32,
            endpoint=True,
        )
        return tuple(values)


class MbaExpr:
    def __init__(self, formula: str) -> None:
        formula = formula.replace("/", "//")
        self.n_vars = _count_variables(formula)
        params = ",".join(f"v{i}" for i in range(self.n_vars))
        self.func: Callable[..., np.uint32] = eval(f"lambda {params}: {formula}")

    def __call__(self, *args: np.uint32) -> np.uint32:
        return self.func(*args)


def example_to_json(inputs: tuple[np.uint32, ...], output: np.uint32) -> dict[str, Any]:
    return {
        "inputs": [{"value": int(x), "size": 32} for x in inputs],
        "output": [{"value": int(output), "size": 32}],
    }


def sample_formula(formula: str, n_examples: int, sampler: Sampler) -> dict[str, Any]:
    expr = MbaExpr(formula)
    examples = []
    for _ in range(n_examples):
        inputs = sampler.gen_values(expr.n_vars)
        outputs = expr(*inputs)
        examples.append((inputs, outputs))

    res = {
        "original_formula": formula,
        "num_variables": expr.n_vars,
        "num_examples": n_examples,
        "examples": [example_to_json(i, o) for i, o in examples],
    }

    return res


def read_file(filepath: Path) -> Iterator[str]:
    with filepath.open() as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                yield line


def write_to_file(json_data: dict[str, Any], out_file: Path) -> None:
    with out_file.open("w+") as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        "-i",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--n_examples",
        "-n",
        type=int,
        required=True,
        help="Number of example inputs to sample for each formula.",
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=Path,
        required=True,
        help="Path of the folder where to write the MBA Learning instances.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Seed for the RNG, to ensure reproducibility. Optional int, defaults to None.",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        default="mba",
        help='Prefix to use for the name of the output files. Optional, defaults to "mba"',
    )

    args = parser.parse_args()
    seed: int = args.seed
    n_examples: int = args.n_examples
    output_folder: Path = args.output_folder
    prefix: str = args.prefix

    output_folder.mkdir(exist_ok=True)
    sampler = Sampler(seed)

    for i, formula in enumerate(read_file(args.input_file)):
        json_data = sample_formula(formula, n_examples, sampler)
        write_to_file(json_data, output_folder / f"{prefix}_{i}_seed={seed}.json")
