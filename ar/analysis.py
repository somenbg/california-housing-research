"""Analysis and visualization of AutoResearch experiment results."""

import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_results(path: str = "results.jsonl") -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_report(results_path: str = "results.jsonl", output_dir: str = "."):
    """Generate a full analysis report: terminal summary + progress.png."""
    records = load_results(results_path)
    if not records:
        print("No experiments found in results.jsonl")
        return

    total = len(records)
    kept = [r for r in records if r["status"] == "keep"]
    discarded = [r for r in records if r["status"] == "discard"]
    crashed = [r for r in records if r["status"] == "crash"]
    direction = records[0].get("metric_direction", "minimize")
    metric_name = records[0].get("metric_name", "val_loss")

    # --- Terminal report ---
    print(f"AutoResearch Analysis Report")
    print(f"{'─' * 50}")
    print()
    print(f"Total experiments:  {total}")
    print(f"  Kept:             {len(kept)}")
    print(f"  Discarded:        {len(discarded)}")
    print(f"  Crashed:          {len(crashed)}")
    if total > 0:
        print(f"  Success rate:     {len(kept)}/{len(kept)+len(discarded)} "
              f"({100*len(kept)/max(1,len(kept)+len(discarded)):.1f}%)")
    print()

    if kept:
        baseline = kept[0]
        if direction == "minimize":
            best = min(kept, key=lambda r: r["metric_value"])
        else:
            best = max(kept, key=lambda r: r["metric_value"])

        bl_val = baseline["metric_value"]
        best_val = best["metric_value"]
        delta = best_val - bl_val
        if bl_val != 0:
            pct = abs(delta / bl_val) * 100
        else:
            pct = 0

        print(f"Metric:             {metric_name} ({'lower' if direction == 'minimize' else 'higher'} is better)")
        print(f"Baseline:           {bl_val:.6f}")
        print(f"Best:               {best_val:.6f}")
        print(f"Total improvement:  {delta:+.6f} ({pct:.1f}%)")
        print()

    # Category breakdown
    categories = Counter()
    category_kept = Counter()
    for r in records:
        cat = r.get("category", "uncategorized") or "uncategorized"
        if r["status"] != "crash":
            categories[cat] += 1
            if r["status"] == "keep":
                category_kept[cat] += 1

    if len(categories) > 1 or (len(categories) == 1 and "baseline" not in categories):
        print("Category breakdown:")
        for cat in sorted(categories, key=categories.get, reverse=True):
            tried = categories[cat]
            success = category_kept.get(cat, 0)
            rate = 100 * success / tried if tried else 0
            print(f"  {cat:24s} {success}/{tried} kept ({rate:.0f}%)")
        print()

    # Improvement velocity
    if len(kept) >= 3:
        print("Improvement velocity:")
        chunk = max(1, total // 3)
        for i, label in enumerate(["First third", "Middle third", "Last third"]):
            start = i * chunk
            end = min((i + 1) * chunk, total) if i < 2 else total
            chunk_records = records[start:end]
            chunk_kept = sum(1 for r in chunk_records if r["status"] == "keep")
            chunk_total = len(chunk_records)
            rate = 100 * chunk_kept / chunk_total if chunk_total else 0
            print(f"  {label:14s} (exp {start+1}-{end}):  {chunk_kept}/{chunk_total} improvements ({rate:.0f}%)")
        print()

    # Top improvements
    if len(kept) > 1:
        print("Kept experiments (improvements):")
        for r in kept:
            desc = r.get("change_summary") or r.get("hypothesis") or ""
            print(f"  #{r['experiment_id']:3d}  {metric_name}={r['metric_value']:.6f}  {desc}")
        print()

    # Platform info
    if records:
        device = records[0].get("platform_device", "unknown")
        device_name = records[0].get("platform_name", "unknown")
        print(f"Platform: {device_name} ({device})")

    # --- Generate chart ---
    chart_path = os.path.join(output_dir, "progress.png")
    _generate_chart(records, metric_name, direction, chart_path)
    print(f"Chart saved to {chart_path}")


def _generate_chart(records: list[dict], metric_name: str, direction: str, output_path: str):
    """Generate progress.png showing experiment results over time."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("AutoResearch Progress", fontsize=16, fontweight="bold")

    # Filter non-crash records for metric plots
    valid = [(i, r) for i, r in enumerate(records) if r["status"] != "crash"]

    if not valid:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    indices = [i + 1 for i, _ in valid]
    values = [r["metric_value"] for _, r in valid]
    statuses = [r["status"] for _, r in valid]

    # --- Plot 1: Metric over experiments ---
    ax1 = axes[0, 0]

    kept_idx = [idx for idx, s in zip(indices, statuses) if s == "keep"]
    kept_val = [v for v, s in zip(values, statuses) if s == "keep"]
    disc_idx = [idx for idx, s in zip(indices, statuses) if s == "discard"]
    disc_val = [v for v, s in zip(values, statuses) if s == "discard"]

    if disc_idx:
        ax1.scatter(disc_idx, disc_val, c="#cccccc", s=20, alpha=0.5,
                    zorder=2, label="Discarded")
    if kept_idx:
        ax1.scatter(kept_idx, kept_val, c="#2ecc71", s=60, zorder=4,
                    label="Kept", edgecolors="black", linewidths=0.5)

    # Running best line
    if kept_val:
        if direction == "minimize":
            running = np.minimum.accumulate(kept_val)
        else:
            running = np.maximum.accumulate(kept_val)
        ax1.step(kept_idx, running, where="post", color="#27ae60",
                 linewidth=2, alpha=0.7, zorder=3, label="Running best")

    ax1.set_xlabel("Experiment #")
    ax1.set_ylabel(metric_name)
    ax1.set_title(f"{metric_name} Over Experiments")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.2)

    # --- Plot 2: Status pie chart ---
    ax2 = axes[0, 1]
    status_counts = {"keep": 0, "discard": 0, "crash": 0}
    for r in records:
        s = r["status"]
        if s in status_counts:
            status_counts[s] += 1

    labels = []
    sizes = []
    colors_map = {"keep": "#2ecc71", "discard": "#e74c3c", "crash": "#95a5a6"}
    for s in ["keep", "discard", "crash"]:
        if status_counts[s] > 0:
            labels.append(f"{s.capitalize()} ({status_counts[s]})")
            sizes.append(status_counts[s])

    if sizes:
        pie_colors = [colors_map.get(s, "#999") for s in ["keep", "discard", "crash"] if status_counts[s] > 0]
        ax2.pie(sizes, labels=labels, colors=pie_colors, autopct="%1.0f%%",
                startangle=90, textprops={"fontsize": 10})
    ax2.set_title("Experiment Outcomes")

    # --- Plot 3: Memory usage over experiments ---
    ax3 = axes[1, 0]
    mem_idx = []
    mem_val = []
    for i, r in enumerate(records):
        sec = r.get("secondary_metrics", {})
        mem = sec.get("peak_memory_mb") or sec.get("peak_vram_mb")
        if mem and r["status"] != "crash":
            mem_idx.append(i + 1)
            mem_val.append(float(mem))

    if mem_idx:
        ax3.bar(mem_idx, mem_val, color="#3498db", alpha=0.7, width=0.8)
        ax3.set_xlabel("Experiment #")
        ax3.set_ylabel("Peak Memory (MB)")
        ax3.set_title("Memory Usage")
        ax3.grid(True, alpha=0.2, axis="y")
    else:
        ax3.text(0.5, 0.5, "No memory data", ha="center", va="center", fontsize=12, color="#999")
        ax3.set_title("Memory Usage")

    # --- Plot 4: Improvement timeline ---
    ax4 = axes[1, 1]
    if len(kept_val) >= 2:
        deltas = []
        for i in range(1, len(kept_val)):
            d = kept_val[i] - kept_val[i - 1]
            deltas.append(d)

        bar_colors = ["#2ecc71" if (d < 0 and direction == "minimize") or
                       (d > 0 and direction == "maximize") else "#e74c3c"
                       for d in deltas]
        ax4.bar(range(1, len(deltas) + 1), deltas, color=bar_colors, alpha=0.8)
        ax4.axhline(y=0, color="black", linewidth=0.5)
        ax4.set_xlabel("Improvement #")
        ax4.set_ylabel(f"Δ {metric_name}")
        ax4.set_title("Per-Improvement Delta")
        ax4.grid(True, alpha=0.2, axis="y")
    else:
        ax4.text(0.5, 0.5, "Need ≥2 kept experiments\nfor delta chart",
                 ha="center", va="center", fontsize=12, color="#999")
        ax4.set_title("Per-Improvement Delta")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
