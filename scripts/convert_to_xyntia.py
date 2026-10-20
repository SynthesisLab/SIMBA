import argparse
import json
import sys
from pathlib import Path


def format_io_pair(inputs, outputs):
    """Helper function to format a single inputs/outputs trace pair."""
    res = {"inputs": {}, "outputs": {}}

    # Format inputs
    for i, inp in enumerate(inputs):
        bit_size = inp["size"]
        mask = (1 << bit_size) - 1  # Ensure unsigned representation
        res["inputs"][str(i)] = {
            "location": f"mem{i}",
            "size": hex(bit_size),  # 32 bits -> "0x20"
            "value": hex(inp["value"] & mask),
        }

    # Format outputs
    for i, out in enumerate(outputs):
        bit_size = out["size"]
        mask = (1 << bit_size) - 1

        # Fallback to generic out0, out1 if there are more outputs than standard registers
        regs = ["EAX", "EBX", "ECX", "EDX"]
        loc = regs[i] if i < len(regs) else f"out{i}"

        res["outputs"][str(i)] = {
            "location": loc,
            "size": hex(bit_size),
            "value": hex(out["value"] & mask),
        }

    return res


def convert_to_xyntia(input_filepath: Path, output_filepath: Path):
    try:
        with input_filepath.open("r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find file {input_filepath}")
        sys.exit(1)

    if "examples" not in data or not data["examples"]:
        print("Error: No examples found in the input JSON.")
        sys.exit(1)

    examples = data["examples"]

    # Xyntia expects the first trace in 'initial' and the rest in 'sampling'
    xyntia_data = {
        "initial": format_io_pair(examples[0]["inputs"], examples[0]["output"]),
        "sampling": {},
    }

    # Map the remaining examples to the 'sampling' dictionary
    for idx, ex in enumerate(examples[1:]):
        xyntia_data["sampling"][str(idx)] = format_io_pair(ex["inputs"], ex["output"])

    # Write to the destination file
    with output_filepath.open("w") as f:
        json.dump(xyntia_data, f, indent=4)

    print(
        f"Successfully converted {input_filepath} to Xyntia format at {output_filepath}"
    )


def process_folder(in_f: Path, out_f: Path):
    out_f.mkdir(parents=True, exist_ok=True)
    for mba in in_f.glob("*.json"):
        out_file = out_f / mba.name
        convert_to_xyntia(mba, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert custom MBA JSON to Xyntia JSON format."
    )
    parser.add_argument(
        "input", type=Path, help="Path to the source JSON file or directory"
    )
    parser.add_argument(
        "output", type=Path, help="Path to the destination JSON file or directory"
    )
    args = parser.parse_args()

    process_folder(args.input, args.output)
