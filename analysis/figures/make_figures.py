"""Regenerate the summary figures in analysis/figures/ from results/.

Run from the repository root:  python analysis/figures/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "results" / "phase1"
OUT = Path(__file__).resolve().parent
MODEL = "Qwen2.5-Omni-3B-Q8_0"

# Categorical slots (fixed order) and text inks.
BLUE, ORANGE, AQUA, YELLOW = "#257de0", "#f47230", "#00b894", "#f2a600"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"

WORKLOADS = {
    0: "chat", 1: "text 512 tok", 2: "text 1,024 tok", 3: "text 3,072 tok",
    4: "1 image", 5: "2 images", 6: "4 images", 7: "audio 10 s", 8: "audio 40 s",
}
# One size per modality, for line charts.
REPR = [(0, "chat", BLUE), (2, "text 1,024 tok", ORANGE), (4, "1 image", AQUA), (7, "audio 10 s", YELLOW)]

CPU_NODES = [  # (results folder label, display name)
    ("c7i.2xlarge", "Intel Sapphire Rapids + AMX\n(c7i.2xlarge)"),
    ("c6i.2xlarge", "Intel Ice Lake, no AMX\n(c6i.2xlarge)"),
    ("c7a.2xlarge", "AMD Genoa\n(c7a.2xlarge)"),
    ("c8g.2xlarge", "ARM Graviton4\n(c8g.2xlarge)"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def v2(label, model=MODEL):
    d = pd.read_csv(P1 / f"{label}__{model}" / "summary_v2.csv")
    d["run"] = d["run"].astype(int)
    return d


def solo(label):
    """Fixed-output, one request at a time, 2 slots; latest row per workload."""
    d = v2(label)
    d = d[(d.output_mode == "fixed") & (d.concurrency == 1) & (d.slots == 2)]
    return d.sort_values("timestamp_utc").groupby("run").last()


def suptitle(fig, title, sub):
    fig.text(0.01, 1.05, title, ha="left", va="bottom", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 1.045, sub, ha="left", va="top", fontsize=9.5, color=INK2)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def node_heatmap():
    """No CPU wins everything: latency relative to the fastest node, per workload."""
    lat = pd.DataFrame({name: solo(lbl)["median_e2e_s"] for lbl, name in CPU_NODES}).T
    rel = lat / lat.min(axis=0)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#eaf2fc", "#0b3f7a"])
    im = ax.imshow(rel.values, cmap=cmap, vmin=1, vmax=rel.values.max(), aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(9), [WORKLOADS[r] for r in rel.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(rel)), rel.index)
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            v = rel.values[i, j]
            ax.text(j, i, "fastest" if v == 1 else f"{v:.2f}×", ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if v == 1 else "normal",
                    color="white" if v > 1 + 0.55 * (rel.values.max() - 1) else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("latency ÷ fastest node", color=INK2)
    cb.outline.set_visible(False)
    suptitle(fig, "No CPU is fastest for every request",
             "End-to-end latency relative to the fastest node, per workload. One request at a time, 128 output tokens, Qwen2.5-Omni-3B Q8_0.")
    save(fig, "node_heatmap.png")


def prefill_decode():
    """Graviton4 decodes ~3x faster but has the slowest long-text prefill."""
    s = {name.replace("\n", " "): solo(lbl) for lbl, name in CPU_NODES}
    names = list(s)
    dec = [s[n].loc[0, "decode_tok_s"] for n in names]
    pre = [s[n].loc[3, "prefill_tok_s"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for ax, vals, title in [(axes[0], dec, "Decode speed, chat (tokens/s)"),
                            (axes[1], pre, "Prefill speed, 3,072-token text (tokens/s)")]:
        best = int(np.argmax(vals))
        ax.barh(names, vals, color=[BLUE if i == best else "#b9cbe3" for i in range(len(vals))],
                height=0.6, edgecolor="white", linewidth=2)
        for i, v in enumerate(vals):
            ax.text(v, i, f"  {v:.0f}", va="center", color=INK, fontsize=9)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.invert_yaxis()
    suptitle(fig, "Decode and prefill favour different CPUs",
             "Decode is memory-bandwidth bound (Graviton4 wins); prefill is compute bound (AMX and Genoa win).")
    save(fig, "prefill_decode.png")


def vcpu_scaling():
    vc = {4: "c7i.1xlarge-q8", 8: "c7i.2xlarge", 16: "c7i.4xlarge-q8"}
    rows = []
    for n, lbl in vc.items():
        d = pd.read_csv(P1 / f"{lbl}__{MODEL}" / "summary.csv")
        d["run"] = d["run"].astype(int)
        d = d.groupby("run").first()
        rows.append((n, d.loc[0, "decode_tok_s"], d.loc[3, "ttft_s"]))
    x, dec, ttft = zip(*rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, y, title, fmt in [(axes[0], dec, "Decode speed, chat (tokens/s)", "{:.1f}"),
                              (axes[1], ttft, "Time to first token, 3,072-token text (s)", "{:.1f} s")]:
        ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=8, mec="white", mew=2)
        for xi, yi in zip(x, y):
            ax.annotate(fmt.format(yi), (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", color=INK, fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [f"{v} vCPUs" for v in x])
        ax.minorticks_off()
        ax.set_ylim(0, max(y) * 1.25)
        ax.set_title(title, loc="left")
    suptitle(fig, "Doubling vCPUs roughly doubles both prefill and decode speed",
             "Intel c7i family (xlarge, 2xlarge, 4xlarge), one request at a time.")
    save(fig, "vcpu_scaling.png")


def label_lines(ax, xs, series):
    """Direct labels at the right end of each line, nudged apart."""
    to_ax = ax.transData + ax.transAxes.inverted()
    ends = sorted([(to_ax.transform((xs[-1], ys[-1]))[1], name) for name, ys, _ in series])
    xa = to_ax.transform((xs[-1], series[0][1][-1]))[0]
    placed = []
    for y, name in ends:
        if placed and y - placed[-1] < 0.065:
            y = placed[-1] + 0.065
        placed.append(y)
        ax.annotate(name, (xa, y), xycoords="axes fraction", xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)


def concurrency():
    d = v2("c7i.2xlarge")
    d = d[(d.output_mode == "fixed") & (d.slots == 2)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    xs = [1, 2, 5, 10]
    for ax, col, title in [(axes[0], "req_s", "Throughput (requests/s)"),
                           (axes[1], "median_e2e_s", "Median end-to-end latency (s)")]:
        series = []
        for run, name, color in REPR:
            g = d[d.run == run].set_index("concurrency").loc[xs, col].values
            ax.plot(xs, g, color=color, lw=2, marker="o", ms=8, mec="white", mew=2)
            series.append((name, g, color))
        ax.axvline(2, color=INK2, lw=1, ls=":")
        ax.text(2, ax.get_ylim()[1], " 2 server slots", color=INK2, fontsize=8.5, va="top")
        ax.set_xticks(xs, [str(v) for v in xs])
        ax.set_xlabel("requests in flight")
        ax.set_title(title, loc="left")
        ax.set_xlim(0.5, 12.5)
        label_lines(ax, xs, series)
    suptitle(fig, "On CPU, concurrency past the slot count only adds queueing",
             "c7i.2xlarge, llama-server with 2 slots. Throughput flattens at 2 in flight while latency keeps growing.")
    save(fig, "cpu_concurrency.png")


def gpu_batching():
    g = pd.read_csv(P1 / f"g6.xlarge-gpu__{MODEL}" / "gpu_summary.csv")
    g = g[g.output_mode == "fixed"].copy()
    g["run"] = g["run"].astype(int)
    g["rerun"] = g.run_id.str.startswith("rerun").astype(int)
    g = g.sort_values(["rerun", "run_id"]).groupby(["slots", "run"]).last().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, col, title in [(axes[0], "out_tok_s", "Output throughput (tokens/s)"),
                           (axes[1], "energy_j_per_request", "GPU energy per request (J)")]:
        series = []
        for run, name, color in REPR:
            s = g[g.run == run].sort_values("slots")
            ax.plot(s.slots, s[col], color=color, lw=2, marker="o", ms=8, mec="white", mew=2)
            series.append((name, s[col].values, color))
        xs = sorted(g.slots.unique())
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(v) for v in xs])
        ax.minorticks_off()
        ax.set_xlabel("batch size (server slots = requests in flight)")
        ax.set_title(title, loc="left")
        ax.set_xlim(xs[0] / 1.3, xs[-1] * 2.6)
        if col == "energy_j_per_request":
            ax.set_yscale("log")
            ax.set_yticks([10, 20, 50, 100], ["10", "20", "50", "100"])
            ax.minorticks_off()
        label_lines(ax, xs, series)
    suptitle(fig, "On GPU, batching keeps paying off",
             "NVIDIA L4 (g6.xlarge), continuous batching in llama-server. Throughput rises and energy per request falls with batch size.")
    save(fig, "gpu_batching.png")


def gpu_vs_cpu():
    cpu = pd.DataFrame({lbl: solo(lbl)["median_e2e_s"] for lbl, _ in CPU_NODES})
    gpu = solo("g6.xlarge-gpu")["median_e2e_s"]
    speed = (cpu.min(axis=1) / gpu).reindex(range(9))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.barh([WORKLOADS[r] for r in speed.index], speed.values, color=BLUE, height=0.6, edgecolor="white", linewidth=2)
    for i, v in enumerate(speed.values):
        ax.text(v, i, f"  {v:.1f}×", va="center", color=INK, fontsize=9)
    ax.axvline(1, color=INK2, lw=1)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, speed.max() * 1.15)
    ax.set_xlabel("GPU speedup over the fastest CPU node (end-to-end latency)")
    suptitle(fig, "One L4 GPU vs. the best CPU for each request",
             "One request at a time, 128 output tokens. The GPU's lead is largest on prefill-heavy requests (long text, images, audio).")
    save(fig, "gpu_vs_cpu.png")


def cost_model():
    r = pd.read_csv(ROOT / "results" / "routing" / "requests.csv")
    r = r[(r.policy == "ect") & (r.status == "ok")]
    mods = [("chat", BLUE), ("text", BLUE), ("image", BLUE), ("audio", BLUE)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharex=True, sharey=True)
    lim = (0.5, 150)
    for ax, (m, c) in zip(axes, mods):
        s = r[r.modality == m]
        ax.plot(lim, lim, color=INK2, lw=1, ls="--")
        ax.scatter(s.predicted_e2e_s, s.e2e_s, s=22, color=c, alpha=0.65, edgecolor="white", linewidth=0.6)
        err = (np.abs(s.predicted_e2e_s - s.e2e_s) / s.e2e_s).median()
        ax.set_title(f"{m}\n{len(s)} requests · median error {err:.0%}", loc="left", fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ticks = [1, 10, 100]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_yticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
    axes[0].set_ylabel("measured (s)")
    for ax in axes:
        ax.set_xlabel("predicted (s)")
    suptitle(fig, "The router's cost model, checked against live runs",
             "Predicted vs. measured end-to-end latency for every request routed by ect, at 60% and 90% load. Dashed line: perfect prediction.")
    save(fig, "cost_model.png")


if __name__ == "__main__":
    node_heatmap()
    prefill_decode()
    vcpu_scaling()
    concurrency()
    gpu_batching()
    gpu_vs_cpu()
    cost_model()
