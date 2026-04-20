import argparse
import json
from pathlib import Path
from typing import Any
import random

import tqdm


_UNARY_OPS = [("-", 1), ("~", 1)]
_BINARY_OPS = [("-", 2), ("+", 2), ("*", 2), ("&", 2), ("|", 2), ("^", 2)]


class Sampler:
    def __init__(self, seed: int | None) -> None:
        self.rng = random.Random(x=seed)

    def random_variable(self, n: int) -> int:
        return self.rng.randrange(n)

    def random_split(self, size: int) -> tuple[int, int | None, str]:
        assert size >= 2, f"random_split called with size = {size}, should be >= 2"
        allowed = []
        if size >= 2:
            allowed += _UNARY_OPS
        if size >= 3:
            allowed += _BINARY_OPS

        op, arity = allowed[self.rng.randrange(len(allowed))]
        if arity == 1:
            return size - 1, None, op
        else:
            left_size = self.rng.randrange(1, size - 1)
            right_size = size - 1 - left_size
            return left_size, right_size, op


def random_formula(size: int, n_variables: int, sampler: Sampler) -> str:
    if size == 1:
        var_id = sampler.random_variable(n_variables)
        return f"v{var_id}"

    left_size, right_size, op = sampler.random_split(size)
    left = random_formula(left_size, n_variables, sampler)
    if right_size is not None:
        right = random_formula(right_size, n_variables, sampler)
        return f"({left} {op} {right})"
    else:
        return f"({op}{left})"


def write_to_file(json_data: dict[str, Any], out_file: Path) -> None:
    with out_file.open("w+") as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size_range",
        "-s",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Size range "
        "(e.g., --size_range 10 20 means all sizes between 10 and 20, inclusive).",
        required=True,
    )
    parser.add_argument(
        "--var_range",
        "-v",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Variable count range"
        " (e.g., --size_range 1 5 means all variable counts between 1 and 5, inclusive).",
        required=True,
    )
    parser.add_argument(
        "--repeats",
        "-r",
        type=int,
        default=1,
        help="Number of times to repeat each size/n_var combination.",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=Path,
        required=True,
        help="Path of the file where to write the MBA formulas.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the RNG, to ensure reproducibility. Optional int, defaults to None.",
    )

    args = parser.parse_args()
    sizes: range = range(args.size_range[0], args.size_range[1] + 1)
    n_vars: range = range(args.var_range[0], args.var_range[1] + 1)
    repeats: int = args.repeats
    output_file: Path = args.output_file
    seed: int = args.seed

    sampler = Sampler(seed)

    with tqdm.tqdm(total=len(sizes) * len(n_vars) * repeats, unit=" formulas") as pbar:
        with output_file.open("w+") as out_f:
            for size in sizes:
                for n_var in n_vars:
                    for r in range(repeats):
                        f = random_formula(size, n_variables=n_var, sampler=sampler)
                        out_f.write(f + "\n")
                        pbar.update()
