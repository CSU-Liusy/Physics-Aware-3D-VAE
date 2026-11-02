"""Build manuscript figures only from hash-checked evaluation outputs.

The --draft flag visibly marks smoke outputs. Formal figures require a complete
three-seed report and a full sealed test. Synthetic examples use predeclared
sample IDs and the central axial slice rather than outcome-driven selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from config_loader import load_config
from experiment_data import ProtocolDataset
from repro_pipeline import stable_hash
from repro_report import _read_run

LABELS = {
    "idw": "IDW", "rbf": "RBF", "ik": "Indicator kriging",
    "mps": "Direct Sampling", "unet": "3D U-Net",
    "base_vae": "Base VAE", "cgan": "Conditional GAN",
    "geovae": "GeoVAE", "geovae_latent": "GeoVAE + latent",
    "geovae_joint": "GeoVAE + latent + LoRA",
}
COLORS = ("#2469a2", "#bf6b25", "#4d8975", "#714f9e", "#555555")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _resolve_manifest(metadata: dict, bundle_root: Path) -> Path:
    """Use only the exact evaluated manifest, including after bundle relocation."""
    expected = metadata["manifest_sha256"]
    original = Path(metadata["manifest"]).resolve()
    candidates = (original, bundle_root / "frozen_inputs" / "manifest.json")
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and _hash(resolved) == expected:
            return resolved
    raise RuntimeError(
        "Evaluated manifest is unavailable or changed; the bundled "
        "frozen_inputs/manifest.json must match the recorded SHA-256")


def _report_includes_evaluation(evaluation_root: Path, report_root: Path,
                                report_meta: dict) -> bool:
    """Match a moved evaluation by its unique path inside the copied bundle."""
    recorded = [Path(path) for path in report_meta["evaluation_directories"]]
    if evaluation_root in [path.resolve() for path in recorded]:
        return True
    try:
        relative = evaluation_root.relative_to(report_root.parent)
    except ValueError:
        return False
    if not relative.parts:
        return False
    matches = [path for path in recorded
               if len(path.parts) >= len(relative.parts)
               and path.parts[-len(relative.parts):] == relative.parts]
    return len(matches) == 1

def require_matplotlib_panel_alignment(fig, axes, report_path: Path,
                                       tolerance_pt: float = 1.5) -> None:
    """Measure final axes and block misaligned comparable rows or columns."""
    fig.canvas.draw()
    matrix = np.asarray(axes, dtype=object)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    width_pt, height_pt = fig.get_size_inches() * 72
    panels = []
    boxes = {}
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axis = matrix[row_index, col_index]
            box = axis.get_position(original=False)
            xy = [float(box.x0 * width_pt), float(box.y0 * height_pt),
                  float(box.x1 * width_pt), float(box.y1 * height_pt)]
            boxes[(row_index, col_index)] = xy
            panels.append({"row": row_index, "column": col_index, "bbox_pt": xy})
    failures = []
    for row_index in range(matrix.shape[0]):
        row_boxes = [boxes[(row_index, j)] for j in range(matrix.shape[1])]
        if max(b[1] for b in row_boxes) - min(b[1] for b in row_boxes) > tolerance_pt:
            failures.append(f"row {row_index} bottom edges differ")
        if max(b[3] for b in row_boxes) - min(b[3] for b in row_boxes) > tolerance_pt:
            failures.append(f"row {row_index} top edges differ")
    for col_index in range(matrix.shape[1]):
        col_boxes = [boxes[(i, col_index)] for i in range(matrix.shape[0])]
        if max(b[0] for b in col_boxes) - min(b[0] for b in col_boxes) > tolerance_pt:
            failures.append(f"column {col_index} left edges differ")
        if max(b[2] for b in col_boxes) - min(b[2] for b in col_boxes) > tolerance_pt:
            failures.append(f"column {col_index} right edges differ")
    report_path.write_text(json.dumps(
        {"tolerance_pt": tolerance_pt, "panels": panels, "failures": failures},
        indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"Figure panel alignment failed: {failures}")


def _export(fig, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.pdf", facecolor="white")
    fig.savefig(output / f"{name}.svg", facecolor="white")
    fig.savefig(output / f"{name}.png", dpi=600, facecolor="white")
    fig.savefig(output / f"{name}.tiff", dpi=600, facecolor="white")


def figures(evaluation: str, report: str, config_path: str, out_dir: str,
            methods: list[str], sample_ids: list[str] | None = None,
            draft: bool = False) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    evaluation_root = Path(evaluation).resolve()
    report_root = Path(report).resolve()
    eval_meta, rows = _read_run(evaluation_root)
    report_path = report_root / "report.json"
    report_meta = json.loads(report_path.read_text(encoding="utf-8"))
    if not draft and report_meta["status"] != "complete_sealed_test":
        raise RuntimeError("Manuscript figures require a complete sealed report")
    if not _report_includes_evaluation(evaluation_root, report_root, report_meta):
        raise RuntimeError("Evaluation is absent from the supplied report")
    if eval_meta["manifest_sha256"] != report_meta["manifest_sha256"]:
        raise RuntimeError("Evaluation/report manifest mismatch")
    if eval_meta["config_sha256"] != report_meta["config_sha256"]:
        raise RuntimeError("Evaluation/report configuration mismatch")
    if eval_meta.get("threshold_selection_sha256") != report_meta["settings"].get("threshold_selection_sha256"):
        raise RuntimeError("Evaluation/report threshold selection mismatch")
    if eval_meta["split"] != report_meta["settings"]["split"]:
        raise RuntimeError("Evaluation/report split mismatch")
    config = load_config(config_path)
    if stable_hash(config) != eval_meta["config_sha256"]:
        raise RuntimeError("Specified configuration differs from evaluated configuration")
    manifest = _resolve_manifest(eval_meta, report_root.parent)
    if len(methods) != len(set(methods)) or len(methods) < 2:
        raise ValueError("Choose two or more unique evaluated methods")
    if not set(methods).issubset(eval_meta["methods"]):
        raise ValueError("Some chosen methods are missing from evaluation")
    if not set(methods).issubset(report_meta["methods"]):
        raise ValueError("Some chosen methods are missing from report")
    run_seed = int(eval_meta["seed"])
    dataset = ProtocolDataset(
        str(manifest), eval_meta["split"],
        num_holes=int(eval_meta["num_holes"]),
        samples_per_hole=int(config["samples_per_hole"]),
        mask_seed=int(eval_meta["mask_seed"]), noise_rate=float(eval_meta["noise_rate"]),
        heldout_holes=4,
    )
    by_key = {(row["sample_id"], row["method"]): row for row in rows}
    case_ids = sorted({row["sample_id"] for row in rows})
    selected = sample_ids or case_ids[:3]
    if len(selected) != 3 or len(set(selected)) != 3:
        raise ValueError("Figure 7 needs exactly three distinct sample IDs")
    if not set(selected).issubset(case_ids):
        raise ValueError("A selected sample ID was not evaluated")
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stamp = "DRAFT CPU SMOKE: NOT FOR MANUSCRIPT" if draft else "Sealed synthetic test"
    sources = {
        "status": "draft" if draft else "sealed",
        "figure_script_sha256": _hash(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
        "selection_rule": ("caller-specified IDs" if sample_ids else
                           "first three sorted IDs, selected without scores"),
        "slice_rule": "sampled depth with largest reference area, for illustration; no model score used",
        "manifest_sha256": _hash(manifest),
        "config_sha256": eval_meta["config_sha256"],
        "evaluation_sha256": _hash(evaluation_root / "evaluation.json"),
        "report_sha256": _hash(report_path),
        "sample_ids": selected, "methods": methods, "seed": run_seed,
        "mask_seed": int(eval_meta["mask_seed"]),
        "prediction_files": [],
    }

    columns = ["Reference", "Observed"] + [LABELS.get(m, m) for m in methods]
    fig, axes = plt.subplots(3, len(columns), figsize=(7.2, 4.25),
                             constrained_layout=True)
    obs_cmap = ListedColormap(["#dddddd", "#2469a2"])
    image = None
    for row_index, sample_id in enumerate(selected):
        case = dataset.get_case(sample_id)
        truth = case["target"]
        observed = case["obs"]
        sampled_depths = np.flatnonzero(np.any(observed[1] >= 0.5, axis=(1, 2)))
        if not len(sampled_depths):
            raise RuntimeError("Selected case has no observed sampled depth")
        z_index = int(sampled_depths[np.argmax(
            truth[sampled_depths].sum(axis=(1, 2)))])
        sources.setdefault("case_slices", []).append(
            {"sample_id": sample_id, "z_index": z_index,
             "target_sha256": dataset.metadata[sample_id]["sha256"]})
        axes[row_index, 0].imshow(truth[z_index], cmap="gray_r",
                                  vmin=0, vmax=1, interpolation="nearest")
        observed_slice = np.ma.masked_where(
            observed[1, z_index] < 0.5, observed[0, z_index])
        axes[row_index, 1].imshow(observed_slice, cmap=obs_cmap,
                                  vmin=0, vmax=1, interpolation="nearest")
        axes[row_index, 1].set_facecolor("white")
        for column_index, method in enumerate(methods, start=2):
            row = by_key[(sample_id, method)]
            source_path = (evaluation_root / row["prediction_file"]).resolve()
            if _hash(source_path) != row["prediction_sha256"]:
                raise RuntimeError("Prediction hash changed during figure creation")
            prediction = np.load(source_path, allow_pickle=False)
            if prediction.shape != truth.shape:
                raise RuntimeError("Prediction and target grid shapes differ")
            image = axes[row_index, column_index].imshow(
                prediction[z_index], cmap="cividis", vmin=0, vmax=1,
                interpolation="nearest")
            sources["prediction_files"].append({
                "sample_id": sample_id, "method": method,
                "path": str(source_path), "sha256": row["prediction_sha256"]})
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].set_ylabel(
            f"{chr(97 + row_index)}  {sample_id}\nz={z_index}",
            fontsize=6, rotation=0, rotation_mode="anchor", labelpad=34,
            ha="right", va="center")
    for index, title in enumerate(columns):
        axes[0, index].set_title(title, fontsize=7)
    if image is not None:
        fig.colorbar(image, ax=axes[:, 2:], location="right", shrink=0.62,
                     label="Ore occupancy probability")
    fig.suptitle(f"{stamp}: reference-rich sampled-depth slices and matched observations",
                 fontsize=8)
    require_matplotlib_panel_alignment(fig, axes, output / "figure7_alignment.json")
    _export(fig, output, "figure7_cases")
    plt.close(fig)

    summary = {row["method"]: row for row in _csv(report_root / "summary.csv")}
    reliability = _csv(report_root / "reliability.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1),
                                   constrained_layout=True)
    y = np.arange(len(methods))
    for index, method in enumerate(methods):
        row = summary[method]
        mean = float(row["blind_iou_mean"])
        low = float(row["blind_iou_ci_low"])
        high = float(row["blind_iou_ci_high"])
        ax1.errorbar(mean, index,
                     xerr=np.array([[max(0, mean - low)], [max(0, high - mean)]]),
                     fmt="o", color=COLORS[index % len(COLORS)],
                     capsize=2, markersize=4)
    ax1.set_yticks(y, [LABELS.get(m, m) for m in methods])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Blind-zone IoU")
    ax1.set_title("a  Parent-bootstrap 95% intervals", loc="left", fontweight="bold")
    ax2.plot((0, 1), (0, 1), "--", color="#555555", linewidth=0.8)
    for index, method in enumerate(methods):
        bins = [row for row in reliability
                if row["method"] == method and int(row["count"]) > 0]
        ax2.plot([float(row["mean_probability"]) for row in bins],
                 [float(row["empirical_frequency"]) for row in bins],
                 "-o", color=COLORS[index % len(COLORS)], markersize=2.5,
                 linewidth=1.0, label=LABELS.get(method, method))
    ax2.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted ore probability",
            ylabel="Observed ore frequency")
    ax2.set_title("b  Blind-voxel reliability", loc="left", fontweight="bold")
    ax2.legend(fontsize=5, loc="lower right")
    fig.suptitle(
        f"{stamp}; {report_meta['case_count']} cases; "
        f"{len(report_meta['seeds'])} training seeds", fontsize=8)
    require_matplotlib_panel_alignment(fig, np.array([[ax1, ax2]], dtype=object),
                                       output / "figure8_alignment.json")
    _export(fig, output, "figure8_performance_calibration")
    plt.close(fig)
    sources["report_data_files"] = {
        name: _hash(report_root / name)
        for name in ("summary.csv", "reliability.csv")}
    (output / "figure_sources.json").write_text(
        json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
    return output / "figure_sources.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--methods", nargs="+", default=["idw", "unet", "geovae"])
    parser.add_argument("--sample-ids", nargs=3)
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    print(figures(args.evaluation, args.report, args.config, args.out,
                  args.methods, args.sample_ids, args.draft))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
