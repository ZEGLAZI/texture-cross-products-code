"""Render the publication's Figure S2 from the calculated validation freeze."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(source, destination):
    records = json.loads(source.read_text())["records"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "axes.titlesize": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 4.6))
    panels = (("H", "Colour–texture"), ("R", "Supervised ResNet-50"),
              ("D", "DINOv2 ViT-S/14"), ("DR", "DINO ResNet-50"),
              ("DV", "DINO ViT-S/16"))
    for ax, (model, title) in zip(axes.flat, panels):
        for code, color, label, style in (("S", "#1c587a", "Squares", "-"),
                                           ("Q", "#bd6b2a", "Full", "--")):
            rows = [r for r in records if r["representation"] == model and r["kernel"] == code]
            values = np.array([[100*c["macro_recall_fraction"] for c in row["curve"]
                                if -12 <= c["j"] <= 8] for row in rows])
            if values.shape != (10, 21):
                raise ValueError(f"Incomplete common validation grid: {model} {code}")
            penalties = 10. ** (np.arange(-12, 9)/4)
            ax.plot(penalties, values.T, color=color, lw=.55, ls=style, alpha=.25)
            ax.plot(penalties, values.mean(axis=0), color=color, lw=1.7, ls=style, label=label)
        ax.set_xscale("log")
        ax.set_xticks(10. ** np.arange(-3, 3),
                      labels=[r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1", "10", r"$10^{2}$"])
        ax.axvline(1., color=".45", ls=":", lw=.8)
        ax.grid(axis="y", color=".90", lw=.5)
        ax.set_axisbelow(True)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].axis("off")
    axes.flat[5].legend(handles, labels, loc="center", frameon=False)
    fig.supxlabel("Ridge penalty λ", fontsize=10)
    fig.supylabel("Validation macro recall (%)", fontsize=10)
    fig.tight_layout(pad=1.1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    provenance = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "plotted_values": 2100, "excluded_from_plot": "extension candidates outside the 21 shared values",
                  "style": "squares solid, full dashed, light horizontal grid",
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination.with_suffix(".json").write_text(json.dumps(provenance, indent=2)+"\n")


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=folder / "choices_frozen.json")
    parser.add_argument("--output", type=Path, default=folder / "ridge_validation_publication")
    args = parser.parse_args()
    render(args.input, args.output)
    print(args.output.with_suffix(".pdf"))
