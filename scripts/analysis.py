from collections import defaultdict
import glob
import json
import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

# matplotlib.use("GTK4Cairo")
# matplotlib.use("GTK4Agg")
# matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import pltpublish

import re

DIGITS_REGEX = re.compile(r"\d+")


def compute_size(f: str) -> int:
    f = f.replace("<<", "<")
    f = f.replace(">>", ">")
    for s in "() ":
        f = f.replace(s, "")
    f = DIGITS_REGEX.sub("", f)

    return len(f)


def parse_smallest_known(file: str) -> str:
    # file = file.replace("tests/", "test2/")
    with open(file, "r") as f:
        data = json.load(f)

    f = data["original_formula"]
    return f


def build_smallest_known() -> dict:
    res = {}
    for file in glob.glob("more_tests/*.json"):
        # remote_name = file.replace("more_tests/", "tests/")
        remote_name = file
        res[remote_name] = parse_smallest_known(remote_name)

    return res


smallest_known = build_smallest_known()


def build_df(fname: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(fname, names=["file", "formula", "time"])
    df["algo"] = name

    df["size"] = df["formula"].map(compute_size)
    df["smallest_known"] = df["file"].map(smallest_known)
    df["smallest_known_size"] = df["smallest_known"].map(compute_size)
    df = df[df["smallest_known_size"] > 10]
    df = df[~df["formula"].str.contains("Not")]
    df = df[~df["formula"].str.contains("Error")]
    df = df.sort_values(by="time")
    df["n_solved"] = range(1, len(df["time"]) + 1)
    df["percent_solved"] = 100 * df["n_solved"] / len(smallest_known)

    return df


def create_plot(a100: bool = False, log_scale: bool = True, percentage: bool = False):
    suffix = "_a100" if a100 else ""
    df_full = build_df(f"../res_fc{suffix}.csv", "full_cache")
    df_light = build_df(f"../res_lc{suffix}.csv", "light_cache")
    df_no = build_df(f"../res_nc{suffix}.csv", "no_cache")

    df = pd.concat([df_no, df_light, df_full])
    print(df.head())

    pltpublish.setup()

    fig, ax = plt.subplots(figsize=(12, 7))

    plt.grid()
    sns.lineplot(
        ax=ax,
        data=df,
        x="time",
        y="percent_solved" if percentage else "n_solved",
        hue="algo",
        style="algo",
        markers=False,
        dashes=False,
    )

    ax.set(
        title=("Percentage" if percentage else "Number")
        + " of instances solved at each time threshold "
        + ("(A100)" if a100 else "(H100)"),
        xlabel="Time (seconds)",
        ylabel="# of instances solved (%)" if percentage else "# of instances solved",
    )

    ax.set_ylim(top=35 if percentage else 750)
    if log_scale:
        ax.set_xscale("log")
    filename = "res_a100.pdf" if a100 else "res_h100.pdf"
    plt.savefig(filename)


def non_binned_plot(a100: bool = False):
    suffix = "_a100" if a100 else ""
    df_full = build_df(f"../res_fc{suffix}.csv", "full_cache")
    df_light = build_df(f"../res_lc{suffix}.csv", "light_cache")
    df_no = build_df(f"../res_nc{suffix}.csv", "no_cache")

    TIMEOUT = 100
    pltpublish.setup()

    time_dict = {row.file: row.time for row in df_no.itertuples()}
    for df2, label in [(df_full, "Full"), (df_light, "Light")]:
        time_dict2 = {row.file: row.time for row in df2.itertuples()}
        xs = []
        ys = []
        for instance in smallest_known:
            t1 = time_dict[instance] if instance in time_dict else TIMEOUT
            ys.append(t1)
            t2 = time_dict2[instance] if instance in time_dict2 else TIMEOUT
            xs.append(t2)

        plt.figure(figsize=(12, 7))
        plt.grid()
        plt.scatter(xs, ys, marker="o", alpha=0.3)
        plt.xlabel(f"Time taken by {label} Cache")
        plt.ylabel("Time taken by No Cache")
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"scatter_nc_vs_{label}{suffix}.pdf")


def binned_plot(a100: bool = False):
    suffix = "_a100" if a100 else ""
    df_full = build_df(f"../res_fc{suffix}.csv", "full_cache")
    df_light = build_df(f"../res_lc{suffix}.csv", "light_cache")
    df_no = build_df(f"../res_nc{suffix}.csv", "no_cache")

    pltpublish.setup()
    time_dict = {row.file: row.time for row in df_no.itertuples()}
    for df2, label in [(df_full, "Full"), (df_light, "Light")]:
        time_dict2 = {row.file: row.time for row in df2.itertuples()}
        counts = defaultdict(int)
        for instance in smallest_known:
            t1 = (
                math.ceil(math.log2(time_dict[instance]))
                if instance in time_dict
                else None
            )
            t2 = (
                math.ceil(math.log2(time_dict2[instance]))
                if instance in time_dict2
                else None
            )
            counts[(t1, t2)] += 1

        print(counts)
        U = -1
        N = math.ceil(math.log2(60)) + 1
        labels = [f"({2 ** (i - 1)}, {2**i}]" for i in range(U, N)] + ["Not solved"]
        data = [
            [counts[(i, j)] for j in range(U, N)] + [counts[(i, None)]]
            for i in range(U, N)
        ] + [[counts[(None, j)] for j in range(U, N)] + [0]]
        a = np.array(data)

        plt.figure(figsize=(12, 7))
        ax = sns.heatmap(
            a,
            annot=True,
            fmt="d",
            linewidths=0.1,
            xticklabels=labels,
            yticklabels=labels,
            cmap="OrRd",
            # cmap="YlOrRd",
            mask=a == 0,
        )
        gpu = "A100" if a100 else "H100"
        ax.invert_yaxis()
        plt.xlabel(f"Time taken by {label} Cache (s)")
        plt.ylabel("Time taken by No Cache (s)")
        plt.title(
            "Comparison of the number of instances solved\n"
            f"by No cache and {label} cache in different time windows "
            f"({gpu} GPU).",
        )
        plt.tight_layout(pad=0.4)
        plt.savefig(f"binned_nc_vs_{label}{suffix}.pdf")
        # plt.show()


if __name__ == "__main__":
    # create_plot(False, log_scale=True, percentage=False)
    # create_plot(True, log_scale=True, percentage=False)
    # non_binned_plot(False)
    # non_binned_plot(True)
    binned_plot(True)
    binned_plot(False)
    # print(len(smallest_known))
