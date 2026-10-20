import argparse
import json
import time
from pathlib import Path
from sys import stderr

import cvc5
from cvc5 import Kind

Input = list[int]
Output = int
Example = tuple[Input, Output]
Task = list[Example]


def solve_cvc5(examples: Task, n_vars: int, bit_vec_size: int = 32) -> str:
    solver = cvc5.Solver()

    # required options
    solver.setOption("sygus", "true")
    solver.setOption("incremental", "false")
    # slv.setOption("verbosity", "3")

    solver.setLogic("QF_BV")

    bit_vec = solver.mkBitVectorSort(bit_vec_size)
    vars = [solver.mkVar(bit_vec, f"v{i}") for i in range(n_vars)]

    # Declare Grammar non-terminals
    start = solver.mkVar(bit_vec, "Start")

    ## Constants
    # zero = slv.mkBitVector(bit_vec_size, 0)
    # one = slv.mkBitVector(bit_vec_size, 1)

    ## Rules
    # https://cvc5.github.io/docs/cvc5-1.2.1/api/python/base/kind.html
    # maybe useful : https://github.com/cvc5/artifact-fmcad23-sygus/blob/main/grammars/bv.sy
    And = solver.mkTerm(Kind.BITVECTOR_AND, start, start)
    Or = solver.mkTerm(Kind.BITVECTOR_OR, start, start)
    Xor = solver.mkTerm(Kind.BITVECTOR_XOR, start, start)
    Not = solver.mkTerm(Kind.BITVECTOR_NOT, start)
    Add = solver.mkTerm(Kind.BITVECTOR_ADD, start, start)
    Sub = solver.mkTerm(Kind.BITVECTOR_SUB, start, start)
    Mul = solver.mkTerm(Kind.BITVECTOR_MULT, start, start)
    Neg = solver.mkTerm(Kind.BITVECTOR_NEG, start)

    # Todo: also allow shifts, etc

    g = solver.mkGrammar(vars, [start])
    g.addRules(
        start,
        [
            # zero, one,
            And,
            Or,
            Xor,
            Not,
            Add,
            Sub,
            Mul,
            Neg,
        ]
        + vars,
    )

    func = solver.synthFun("f", vars, bit_vec, g)
    ## Add constraints from examples.
    # For each example pair
    for inputs, output in examples:
        # Create values from input variables
        input_bitvecs = [solver.mkBitVector(bit_vec_size, x) for x in inputs]
        # Apply f to these values
        f_output = solver.mkTerm(Kind.APPLY_UF, func, *input_bitvecs)
        # Create value for output
        output_bitvec = solver.mkBitVector(bit_vec_size, output)
        # Constraint: value of f must be equal to value of output.
        solver.addSygusConstraint(solver.mkTerm(Kind.EQUAL, f_output, output_bitvec))

    print("starting synth", file=stderr)
    if solver.checkSynth().hasSolution():
        print("Solution found:", file=stderr)
        terms = [func]
        formula = rebuild_formula(solver.getSynthSolutions(terms)[0])
        return formula
    else:
        print("No solution found", file=stderr)
        return "Not found!"


kind_to_op = {
    Kind.BITVECTOR_AND: "&",
    Kind.BITVECTOR_OR: "|",
    Kind.BITVECTOR_XOR: "^",
    Kind.BITVECTOR_ADD: "+",
    Kind.BITVECTOR_SUB: "-",
    Kind.BITVECTOR_MULT: "*",
    Kind.BITVECTOR_NOT: "~",
    Kind.BITVECTOR_NEG: "-",
}


def rebuild_formula(term: cvc5.Term) -> str:
    kind = term.getKind()
    match kind:
        case Kind.LAMBDA:
            return rebuild_formula(term[1])
        case (
            Kind.BITVECTOR_AND
            | Kind.BITVECTOR_OR
            | Kind.BITVECTOR_XOR
            | Kind.BITVECTOR_ADD
            | Kind.BITVECTOR_SUB
            | Kind.BITVECTOR_MULT
        ):
            left = rebuild_formula(term[0])
            right = rebuild_formula(term[1])
            op = kind_to_op[kind]
            return f"({left} {op} {right})"
        case Kind.BITVECTOR_NOT | Kind.BITVECTOR_NEG:
            child = rebuild_formula(term[0])
            op = kind_to_op[kind]
            return f"({op}{child})"
        case Kind.VARIABLE:
            return term.getSymbol()
        case _:
            raise RuntimeError(f"Unsupported kind: {kind}")


def examples_from_file(file: Path) -> tuple[Task, int]:
    with file.open("r") as f:
        data = json.load(f)
    n_vars, n_examples = data["num_variables"], data["num_examples"]
    task = []
    for i in range(n_examples):
        input = [x["value"] for x in data["examples"][i]["inputs"]]
        output = data["examples"][i]["output"][0]["value"]
        example = (input, output)
        task.append(example)
    return task, n_vars


if __name__ == "__main__":
    N_BITS = 32
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    ex, n_vars = examples_from_file(args.file)
    start = time.time()
    formula = solve_cvc5(ex, n_vars, N_BITS)
    elapsed = time.time() - start
    print(f"{args.file},{formula},{elapsed:.3f}")
