"""Vector figures from the synthetic revision evidence; never reads real meshes.

Run with --schematics before results exist. Quantitative assets are generated
only when their actual source tables are available. No placeholder numbers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import numpy as np

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
                     "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE = "#4477AA"
GREEN = "#228833"
GRAY = "#666666"
PALE = "#EDF3F8"
# The submitted sn-jnl one-column layout has a 31 pc text block.  Draw at
# its physical width so LaTeX does not silently reduce 6 pt labels to 4.5 pt.
WIDTH_MM = 130.75
WIDTH = WIDTH_MM / 25.4


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(fig, out, name, contract, sources=(), *, alignment_options=None, draft=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if draft:
        fig.text(.5, .985, "CPU SMOKE DRAFT | functional checks only; not manuscript evidence",
                 ha="center", va="top", fontsize=7, color="#883344")
    fig.canvas.draw()
    # The optional external QA tool inspects geometry; it does not draw assets.
    auditor = os.environ.get("GEOVAE_FIGURE_ALIGNMENT_AUDITOR")
    if auditor:
        spec = importlib.util.spec_from_file_location("figure_alignment_qa", auditor)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        options = dict(alignment_options or {})
        if not options and len(fig.axes) > 1:
            # The original schematic uses manually positioned axes in one row.
            # New data figures supply their real row/column comparison groups.
            options["row_groups"] = [list("abcdefghijklmnopqrstuvwxyz"[:len(fig.axes)])]
        alignment = module.require_matplotlib_panel_alignment(
            fig, json_out=out / (name + ".alignment.json"),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
            **options)
    else:
        alignment = {"status": "NOT AUDITABLE", "reason": "External alignment audit not supplied"}
        _write_json(out / (name + ".alignment.json"), alignment)
    fig.savefig(out / (name + ".pdf"))
    fig.savefig(out / (name + ".svg"))
    fig.savefig(out / (name + ".png"), dpi=300)
    font_audit = {"status": "NOT AUDITABLE", "reason": "External PDF glyph audit not supplied"}
    font_tool = os.environ.get("GEOVAE_FIGURE_FONT_AUDITOR")
    if font_tool:
        spec = importlib.util.spec_from_file_location("figure_font_qa", font_tool)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        font_audit = module.audit_pdf((out / (name + ".pdf")).read_bytes(), minimum_pt=5.)
        font_audit["status"] = ("PASS" if font_audit["auditable"] and not font_audit["below_minimum_count"]
                                else "FAIL" if font_audit["auditable"] else "NOT AUDITABLE")
        _write_json(out / (name + ".fonts.json"), font_audit)
        if font_audit["status"] != "PASS":
            raise RuntimeError("Final-size PDF glyph audit did not pass: " + name)
    collision = {"status": "NOT AUDITABLE", "reason": "External PDF collision audit not supplied"}
    collision_tool = os.environ.get("GEOVAE_FIGURE_COLLISION_AUDITOR")
    if collision_tool:
        qa_python = os.environ.get("GEOVAE_FIGURE_QA_PYTHON", sys.executable)
        result = subprocess.run([qa_python, "-X", "utf8", collision_tool, str(out / (name + ".pdf")),
                                 "--json-out", str(out / (name + ".collision.json"))],
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError("Rendered figure collision audit failed: " + result.stdout + result.stderr)
        collision = json.loads((out / (name + ".collision.json")).read_text(encoding="utf-8"))
        # Audit geometry is retained; local output locations are not public provenance.
        collision["pdf"] = name + ".pdf"
        _write_json(out / (name + ".collision.json"), collision)
    _write_json(out / (name + ".provenance.json"), {
        "figure": name, "contract": contract, "data_policy": "synthetic_only",
        "sources": [{"file": Path(p).name, "sha256": _hash(p)} for p in sources],
        "script_sha256": _hash(__file__), "alignment_audit": alignment,
        "collision_audit": collision, "evidence_role": "cpu_smoke_draft" if draft else "protocol_or_actual_experiment",
        "export_width_mm": float(fig.get_figwidth() * 25.4),
        "export_height_mm": float(fig.get_figheight() * 25.4),
        "target_textwidth_mm": WIDTH_MM, "glyph_audit": font_audit,
        "embedding_policy": "Use the native width; any height cap must not shrink the verified glyph floor below 5 pt.",
        "note": ("Illustrative arrays and schematic labels are protocol illustrations, not measured performance."
                 if contract.get("data_type", "").startswith("illustrative") or "schematic" in contract.get("archetype", "")
                 else "Actual saved synthetic outputs only; undefined quantities are not imputed.")})
    plt.close(fig)


def _box(ax, xy, width, height, title, body, color=BLUE):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                 boxstyle="round,pad=0.008,rounding_size=0.012",
                 facecolor=PALE, edgecolor=color, linewidth=0.8))
    ax.text(x + width / 2, y + height * .76, title, ha="center", va="center",
            fontsize=7.2, color=color, fontweight="bold", linespacing=1.15)
    ax.text(x + width / 2, y + height * .30, body, ha="center", va="center",
            fontsize=6.6, linespacing=1.20)


def _arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                linewidth=.9, color=GRAY))


def schematics(out):
    fig, ax = plt.subplots(figsize=(WIDTH, 75 / 25.4))
    fig.subplots_adjust(left=.02, right=.98, bottom=.09, top=.97)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    titles = ["Independent synthetic\nparents", "Parent-level split", "Fixed sparse\nobservations",
              "Train on complete\ntargets", "Select on validation\nonly", "Evaluate frozen\nsettings"]
    bodies = ["1,000 generating parents\n5 variants per parent",
              "700 / 150 / 150 parents\n3,500 / 750 / 750 volumes",
              "Eight straight columns\n12 depths; identical masks",
              "Two-channel sparse input\nComplete training target",
              "Predeclared candidates\nFreeze before new testing",
              "Blind and observed errors\nCalibration, failures, runtime"]
    for i in range(6):
        row, col = divmod(i, 3)
        _box(ax, (.025 + col * .325, .59 if row == 0 else .13), .295, .30,
             titles[i], bodies[i])
    for y in (.74, .28):
        _arrow(ax, (.325, y), (.345, y)); _arrow(ax, (.650, y), (.670, y))
    ax.plot([.975, .990, .990, .010, .010], [.74, .74, .55, .55, .28],
            color=GRAY, linewidth=.8)
    _arrow(ax, (.010, .28), (.025, .28))
    ax.text(.5, .49, "No test labels or real segments enter training or selection",
            ha="center", va="center", fontsize=7.5, color=GRAY)
    ax.text(.5, .025, "All geometric illustrations and primary evidence are synthetic",
            ha="center", va="center", fontsize=7.5)
    export(fig, out, "workflow", {"question": "How are data and selection isolated?",
           "claim": "Parent-disjoint partitions and validation-only selection define the revised benchmark.",
           "archetype": "schematic-led composite"})

    # A deliberately illustrative target, with identical sample coordinates in
    # all panels. It is not an experimental reconstruction or a real orebody.
    z, x = np.indices((32, 32))
    truth = ((((x - 16) / 10) ** 2 + ((z - 16) / 7) ** 2) < 1).astype(float)
    mask = np.zeros_like(truth)
    for column in (7, 16, 25):
        mask[np.rint(np.linspace(0, 31, 12)).astype(int), column] = 1
    observed = truth * mask
    source_data = Path(out) / "source_data"
    source_data.mkdir(parents=True, exist_ok=True)
    encoding_source = source_data / "input_encoding.npz"
    np.savez_compressed(encoding_source, target=truth, observed_labels=observed,
                        observation_mask=mask, sample_coordinates_zx=np.argwhere(mask > .5))
    fig = plt.figure(figsize=(WIDTH, 65 / 25.4))
    axes = [fig.add_axes([.09 + i * .30, .35, .24, .54]) for i in range(3)]
    cmap = ListedColormap(["#F4F4F4", BLUE])
    for i, (ax, image, title) in enumerate(zip(axes, (truth, observed, mask),
             ("Complete target V", "Sparse labels Vobs", "Observation mask M"))):
        ax.imshow(image, cmap=cmap, origin="lower", vmin=0, vmax=1,
                  extent=(-.5, 31.5, -.5, 31.5), interpolation="nearest")
        zs, xs = np.where(mask > .5)
        ax.scatter(xs, zs, s=5, facecolors="none", edgecolors="#222222", linewidths=.35)
        ax.set_xticks([0, 16, 31]); ax.set_yticks([0, 16, 31])
        ax.set_title(title, fontsize=8.3, pad=7)
        ax.set_xlabel("x index")
        ax.text(-.10, 1.08, chr(97+i), transform=ax.transAxes, fontweight="bold", fontsize=9)
        ax.set_ylabel("z index" if i == 0 else "")
    fig.text(.5, .10, "(Vobs, M): unknown (0, 0); sampled barren (0, 1); sampled occupied (1, 1)",
             ha="center", fontsize=6.6)
    fig.text(.5, .025, "Illustrative synthetic section; identical sample positions in every panel",
             ha="center", fontsize=6.6)
    export(fig, out, "input_encoding", {"question": "How is unknown space distinguished from sampled barren space?",
           "claim": "The second channel distinguishes labels that are zero for different reasons.",
           "archetype": "quantitative grid", "data_type": "illustrative_binary_array",
           "columns": [7, 16, 25], "integrity": "The displayed labels, mask and exact sample coordinates are archived together."},
           [encoding_source])

    fig, ax = plt.subplots(figsize=(WIDTH, 75 / 25.4))
    fig.subplots_adjust(left=.015, right=.985, top=.97, bottom=.04)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    upper = [("Input", "2 x 32 x 32 x 32"), ("Encoder", "16 / 32 / 64 /\n128 / 256 channels"),
             ("Attention", "8 tokens; 4 heads"), ("Latent heads", "Mean and\nlog variance")]
    lower = [("Occupancy", "Sigmoid\n32 x 32 x 32"), ("Decoder", "4 trilinear\nupsampling stages"),
             ("Attention", "8 tokens\nsigned offsets"), ("Latent vector", "256 dimensions")]
    for y, items in ((.66, upper), (.17, lower)):
        for i, (title, body) in enumerate(items):
            _box(ax, (.015+i*.25, y), .22, .23, title, body)
    for i in range(3):
        _arrow(ax, (.240+i*.25, .775), (.260+i*.25, .775))
        _arrow(ax, (.260+i*.25, .285), (.240+i*.25, .285))
    _arrow(ax, (.875, .645), (.875, .415))
    ax.text(.47, .52, "Residual convolutions, channel weighting and group normalization",
            ha="center", fontsize=7.1)
    ax.text(.47, .455, "Relative bias = Bx + By + Bz, indexed by signed token offsets",
            ha="center", fontsize=7.1)
    ax.text(.5, .045, "Case-specific inference: no adaptation / latent only / latent + decoder LoRA",
            ha="center", fontsize=6.8)
    export(fig, out, "architecture", {"question": "Which representation and adaptation are actually implemented?",
           "claim": "The implemented model uses bottleneck relative-position attention and a 256-dimensional latent vector.",
           "archetype": "schematic-led composite"})


DISPLAY = {"idw": "IDW", "rbf": "RBF", "ik": "Indicator kriging", "mps": "MPS (GENESIM)",
           "unet": "3D U-Net", "base_vae": "Basic VAE", "cgan": "3D cGAN",
           "geovae": "GeoVAE", "geovae_latent": "GeoVAE + z", "geovae_joint": "GeoVAE + z + LoRA"}
COLORS = {"idw": "#777777", "rbf": "#AA7744", "ik": "#999966", "mps": "#998899",
          "unet": "#CC9955", "base_vae": "#669988", "cgan": "#AA77AA",
          "geovae": "#4477AA", "geovae_latent": "#66AACC", "geovae_joint": "#224466"}
MARKERS = dict(zip(DISPLAY, ("o", "s", "^", "D", "s", "D", "v", "o", "^", "P")))


class EvidenceUnavailable(RuntimeError):
    """A completed source is absent; no quantitative substitute is generated."""


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise EvidenceUnavailable("Required source is absent: " + path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path):
    path = Path(path)
    if not path.is_file():
        raise EvidenceUnavailable("Required table is absent: " + path.name)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise EvidenceUnavailable("Required table has no completed rows: " + path.name)
    return rows


def _number(value):
    if value in (None, "", "None", "null"):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite figure source value")
    return number


def _statistics(row, metric):
    """Accept the documented long schema and the engine's wide schema."""
    names = ("mean", "ci_low", "ci_high") if row.get("metric") == metric else tuple(
        metric + "_" + suffix for suffix in ("mean", "ci_low", "ci_high"))
    values = [_number(row.get(key)) for key in names]
    if any(value is None for value in values):
        raise EvidenceUnavailable("Required metric/interval is absent: " + metric)
    mean, low, high = values
    if low > mean + 1e-10 or high < mean - 1e-10:
        raise ValueError("Interval does not contain its reported mean")
    return mean, max(0., mean - low), max(0., high - mean)


def _metric_row(rows, method, metric):
    matches = [row for row in rows if row["method"] == method
               and ("metric" not in row or row["metric"] == metric)]
    if len(matches) != 1:
        raise EvidenceUnavailable("Expected one method/metric summary: " + method + "/" + metric)
    return matches[0]


def _methods(rows, draft=False):
    actual = {row["method"] for row in rows}
    if actual - set(DISPLAY):
        raise ValueError("Unregistered method in figure source")
    if not draft and actual != set(DISPLAY):
        raise EvidenceUnavailable("Formal main comparison requires all ten registered methods")
    return [method for method in DISPLAY if method in actual]


def _panel_labels(axes):
    for index, ax in enumerate(axes):
        ax.annotate(chr(97 + index), xy=(0, 1), xycoords="axes fraction",
                    xytext=(-11, 7), textcoords="offset points", ha="right", va="bottom",
                    fontsize=8, fontweight="bold", annotation_clip=False)


def _alignment(axes, ncols, *, exclude=()):
    ids = [chr(97 + index) for index in range(len(axes))]
    row_groups = [ids[i:i + ncols] for i in range(0, len(ids), ncols)
                  if len(ids[i:i + ncols]) > 1]
    column_groups = [ids[i::ncols] for i in range(ncols) if len(ids[i::ncols]) > 1]
    return {"panel_ids": dict(zip(axes, ids)), "row_groups": row_groups,
            "column_groups": column_groups, "exclude_axes": list(exclude)}


def _source_table(out, name, rows):
    folder = Path(out) / "source_data"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + ".csv")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _synthetic_manifest(run, explicit=None):
    candidates = [Path(explicit)] if explicit else [run / "frozen_inputs" / "manifest.json",
                                                   run / "protocol" / "manifest.json"]
    path = next((path for path in candidates if path.is_file()), None)
    if path is None:
        raise EvidenceUnavailable("A synthetic protocol manifest is required for quantitative figures")
    manifest = _read(path)
    if (manifest.get("protocol_version") != "procedural-lens-v1"
            or manifest.get("source_data_policy") != "independent_procedural_only_no_real_mesh_parents"
            or any(row.get("source") != "independent_procedural_lens" or row.get("source_parent") is not None
                   for row in manifest.get("samples", []))):
        raise ValueError("Figure input must be the independent synthetic protocol; real meshes are forbidden")
    return path, manifest


def _analysis(run, manifest, draft):
    folder = run / "revision_analysis"
    if not folder.is_dir() and draft:
        folder = run / "analysis"
    meta = _read(folder / "analysis.json")
    if meta.get("manifest_sha256") != _hash(manifest):
        raise ValueError("Analysis and plotting manifest identities differ")
    if not draft and meta.get("status") != "complete_primary_reanalysis":
        raise EvidenceUnavailable("Formal figure requires a complete primary reanalysis")
    return folder, meta


def _guard_public_output(run, out, manifest, draft):
    """Gate public synthetic figures on the actual completed handoff and its inputs.

    This is a public-copy patch. The author-side submission gate below retains its
    original semantics. Metadata and hashes do not re-execute the experiments.
    """
    run, manifest = Path(run), Path(manifest)
    root = Path(__file__).resolve().parent
    in_submission_tree = "revision_article" in {part.lower() for part in Path(out).resolve().parts}
    if draft or (run / "smoke_identity.json").exists():
        if in_submission_tree:
            raise ValueError("CPU smoke/draft quantitative figures may only be written to an internal QA destination")
        return
    provenance_path = root / "EXPORT_PROVENANCE.json"
    if not in_submission_tree and not provenance_path.is_file():
        if ((run / "run_identity.json").is_file()
                and _read(run / "run_identity.json").get("public_mode") == "synthetic_only"):
            raise ValueError("Public synthetic figures require the export provenance record")
        return
    identity = _read(run / "run_identity.json")
    if identity.get("public_mode") == "synthetic_only" and not provenance_path.is_file():
        raise ValueError("Public synthetic figures require the export provenance record")
    status = _read(run / "revision_status.json")
    _, analysis = _analysis(run, manifest, False)
    trained = _read(run / "models" / "geovae" / "seed11" / "run.json")
    config = trained.get("config", {})
    formal_shape = (
        identity.get("mode") == "formal"
        and int(config.get("latent_dim", -1)) == 256
        and config.get("grid_size") == [32, 32, 32]
        and analysis.get("case_count") == 750
        and sorted(analysis.get("seeds", [])) == [11, 23, 37]
        and set(analysis.get("methods", [])) == set(DISPLAY)
        and trained.get("manifest_sha256") == _hash(manifest)
    )
    if provenance_path.is_file():
        provenance = _read(provenance_path)
        handoff = _read(run / "public_handoff.json")
        manifest_sha = _hash(manifest)
        core = ("repro_pipeline.py", "experiment_data.py", "model.py", "repro_cgan.py",
                "repro_baselines.py", "repro_metrics.py", "comparisons/models/unet.py")
        digest = hashlib.sha256()
        for name in core:
            path = root / name
            if not path.is_file():
                raise ValueError("A checkpoint-identity engine file is missing: " + name)
            file_sha = _hash(path)
            if (identity.get("code_files", {}).get(name) != file_sha
                    or provenance.get("checkpoint_engine_files", {}).get(name) != file_sha):
                raise ValueError("Public figure engine differs from the frozen run: " + name)
            digest.update(name.encode("utf-8"))
            digest.update(file_sha.encode("ascii"))
        engine_sha = digest.hexdigest()
        policy = handoff.get("evaluation_numerical_policy_audit", {})
        numerics_path = root / "revision_numerics.py"
        if not numerics_path.is_file():
            raise ValueError("Public numerical policy helper is missing")
        numerics_sha = _hash(numerics_path)
        protocol = _read(manifest)
        if not (
            formal_shape
            and identity.get("public_mode") == "synthetic_only"
            and identity.get("data_root") == "not_available_restricted"
            and identity.get("raw_inventory_sha256") is None
            and identity.get("raw_audit_sha256") is None
            and status.get("status") == "public_synthetic_experiments_complete"
            and handoff.get("status") == "public_synthetic_reproduction_complete"
            and handoff.get("manuscript_ready") is False
            and handoff.get("raw_real_audit_passed") is False
            and handoff.get("verified_robustness_predictions", 0) > 0
            and provenance.get("engine_source_sha256") == engine_sha
            and identity.get("source_sha256") == engine_sha
            and handoff.get("source_sha256") == engine_sha
            and trained.get("source_sha256") == engine_sha
            and analysis.get("source_sha256") == engine_sha
            and provenance.get("synthetic_manifest_sha256") == manifest_sha
            and identity.get("manifest_sha256") == manifest_sha
            and handoff.get("manifest_sha256") == manifest_sha
            and analysis.get("manifest_sha256") == manifest_sha
            and protocol.get("protocol_version") == "procedural-lens-v1"
            and protocol.get("count") == 5000
            and protocol.get("grid_size") == [32, 32, 32]
            and protocol.get("split_counts") == {"train": 3500, "val": 750, "test": 750}
            and protocol.get("parent_counts") == {"train": 700, "val": 150, "test": 150}
            and sorted(identity.get("seeds", [])) == [11, 23, 37]
            and set(identity.get("methods", [])) == set(DISPLAY)
            and trained.get("seed") == 11
            and trained.get("kind") == "geovae"
            and analysis.get("analysis_script_sha256") == _hash(root / "revision_analysis.py")
            and identity.get("evaluation_numerics_source_sha256") == numerics_sha
            and policy.get("helper_sha256") == numerics_sha
            and policy.get("evaluation_phase_count", 0) >= 9
            and policy.get("training_phase_count", 0) >= 1
        ):
            raise ValueError("Public quantitative figures require a verified complete synthetic handoff with matching formal identities")
        from public_finish import figure_evidence_paths
        required = figure_evidence_paths(run)
        for name in required:
            path = run / name
            if (not path.is_file()
                    or handoff.get("key_files", {}).get(name) != _hash(path)):
                raise ValueError("Public handoff result is absent or changed: " + name)
        robust = _read(run / "revision_robustness" / "status.json")
        if robust.get("status") != "complete_subset_diagnostic" or robust.get("case_count") != 30:
            raise ValueError("Public robustness diagnostic is incomplete")
        return
    if not (formal_shape and status.get("status") == "experiments_complete"):
        raise ValueError("Submission quantitative exports require the completed formal d256 run and full 750-case, three-seed primary reanalysis")



def main_comparison(run, out, manifest, draft=False):
    _guard_public_output(run, out, manifest, draft)
    folder, meta = _analysis(run, manifest, draft)
    source = folder / "summary.csv"
    rows = _rows(source)
    methods = _methods(rows, draft)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 85 / 25.4), sharey=True)
    fig.subplots_adjust(left=.245, right=.95, top=.86, bottom=.27, wspace=.22)
    for ax, metric, label in zip(axes, ("blind_iou", "observed_error"),
                                 ("Blind-region IoU", "Observed label error")):
        for i, method in enumerate(methods):
            mean, low, high = _statistics(_metric_row(rows, method, metric), metric)
            ax.errorbar(mean, i, xerr=[[low], [high]], fmt=MARKERS[method], color=COLORS[method],
                        markersize=4, capsize=2, elinewidth=.9)
        ax.set_xlabel(label)
        ax.grid(axis="x", color="#EAEAEA", linewidth=.6)
        ax.set_yticks(np.arange(len(methods)), [DISPLAY[m] for m in methods])
        ax.set_ylim(len(methods) - .5, -.5)
    axes[1].tick_params(labelleft=False)
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1000:g}"))
    axes[1].set_xlabel("Observed label error\n(×10⁻³)")
    _panel_labels(axes)
    parents = sorted({int(row["independent_parents"]) for row in rows
                      if row.get("independent_parents") not in (None, "")})
    parent_text = str(parents[0]) if len(parents) == 1 else "source-table counts"
    fig.text(.5, .075, "Equal-parent means; 95% parent-bootstrap intervals. Parents: " + parent_text,
             ha="center", fontsize=6.7)
    export(fig, out, "main_comparison", {
        "question": "How do blind reconstruction and observed consistency compare?",
        "claim": "Both outcomes are reported for the same registered synthetic cohort, without assuming a method wins.",
        "panels": {"a": "Blind-region reconstruction", "b": "Consistency on observed labels"},
        "archetype": "quantitative grid", "bootstrap": meta.get("bootstrap"),
        "model_seeds": meta.get("seeds"), "methods": methods,
        "test_volumes": meta.get("case_count"),
        "exclusions": "None among registered methods; missing metrics block this figure."},
        [manifest, folder / "analysis.json", source], alignment_options=_alignment(axes, 2), draft=draft)


def calibration(run, out, manifest, draft=False):
    _guard_public_output(run, out, manifest, draft)
    folder, meta = _analysis(run, manifest, draft)
    source = folder / "reliability.csv"
    rows = _rows(source)
    methods = _methods(rows, draft)
    groups = ([method for method in methods if method in ("idw", "rbf", "ik", "mps")],
              [method for method in methods if method not in ("idw", "rbf", "ik", "mps")])
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 85 / 25.4))
    fig.subplots_adjust(left=.115, right=.98, top=.84, bottom=.39, wspace=.31)
    empty_bins, totals = {}, {}
    for ax, group, title in zip(axes, groups, ("Classical methods", "Learned methods")):
        ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle="--", linewidth=.7)
        for method in group:
            selected = sorted([row for row in rows if row["method"] == method], key=lambda row: int(row["bin"]))
            if len({row["bin"] for row in selected}) != len(selected):
                raise ValueError("Duplicate reliability bin")
            valid = [row for row in selected if int(row["prediction_voxel_count"]) > 0]
            empty_bins[method] = len(selected) - len(valid)
            totals[method] = sum(int(row["prediction_voxel_count"]) for row in valid)
            if not valid:
                raise EvidenceUnavailable("No nonempty reliability bin for " + method)
            xs, ys = [], []
            for row in valid:
                x, y = _number(row["mean_probability"]), _number(row["empirical_frequency"])
                if x is None or y is None or not (0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError("Invalid probability/frequency in nonempty reliability bin")
                xs.append(x); ys.append(y)
            ax.plot(xs, ys, marker=MARKERS[method], markersize=3.5, linewidth=.9,
                    color=COLORS[method], label=DISPLAY[method])
        ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), xlabel="Mean predicted occupancy",
               title=title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    axes[0].set_ylabel("Empirical occupancy frequency")
    _panel_labels(axes)
    handles, labels = [], []
    for ax in axes:
        part_handles, part_labels = ax.get_legend_handles_labels()
        handles.extend(part_handles); labels.extend(part_labels)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52, .09), ncol=3,
               fontsize=6.5, frameon=False, columnspacing=1., handlelength=1.5)
    fig.text(.5, .025, f"{meta.get('case_count')} synthetic volumes; {len(meta.get('seeds', []))} learned-model seeds. Empty bins omitted.\n"
             "Voxel counts are descriptive, not independent sample sizes.",
             ha="center", fontsize=6.3, linespacing=1.25)
    export(fig, out, "calibration", {
        "question": "Do forecast probabilities agree with observed blind-voxel frequencies?",
        "claim": "Reliability is assessed separately from shape overlap, using all nonempty stored bins.",
        "archetype": "quantitative grid", "panels": {"a": "Classical probability forecasts", "b": "Learned probability forecasts"},
        "bin_voxel_counts": totals, "zero_count_bins_omitted": empty_bins,
        "test_volumes": meta.get("case_count"), "model_seeds": meta.get("seeds"),
        "calibration_policy": meta.get("calibration_policy"),
        "interval": "None: pooled bin frequencies are descriptive; no independent-voxel confidence intervals are asserted.",
        "transform": "Stored bin means connected for readability; empty-bin values are never imputed."},
        [manifest, folder / "analysis.json", source], alignment_options=_alignment(axes, 2), draft=draft)


def runtime(run, out, manifest, draft=False):
    _guard_public_output(run, out, manifest, draft)
    folder, meta = _analysis(run, manifest, draft)
    source = folder / "summary.csv"
    rows = _rows(source)
    methods = _methods(rows, draft)
    durations, sources = [], [manifest, source, folder / "analysis.json"]
    expected_seeds = set(meta.get("seeds", []))
    for model in ("geovae", "base_vae", "unet", "cgan"):
        histories = sorted((run / "models" / model).glob("seed*/history.jsonl"))
        found = set()
        for path in histories:
            history = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not history or len({int(row["epoch"]) for row in history}) != len(history):
                raise ValueError("Missing or duplicate training epoch history")
            seed = int(path.parent.name.removeprefix("seed"))
            found.add(seed)
            seconds = [_number(row.get("seconds")) for row in history]
            if any(value is None or value < 0 for value in seconds):
                raise ValueError("Training wall-clock history is incomplete")
            durations.append({"model": model, "seed": seed, "recorded_epochs": len(history),
                              "recorded_epoch_seconds_sum": sum(seconds)})
            sources.append(path)
        if found != expected_seeds:
            raise EvidenceUnavailable("Runtime figure requires all registered model-seed histories")
    table = _source_table(out, "training_durations", durations)
    sources.append(table)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 90 / 25.4))
    fig.subplots_adjust(left=.15, right=.98, top=.82, bottom=.25, wspace=.88)
    models = ("geovae", "base_vae", "unet", "cgan")
    unit = 60. if max(row["recorded_epoch_seconds_sum"] for row in durations) >= 300 else 1.
    for i, model in enumerate(models):
        selected = [row for row in durations if row["model"] == model]
        xs = [row["recorded_epoch_seconds_sum"] / unit for row in selected]
        jitter = np.linspace(-.12, .12, len(xs)) if len(xs) > 1 else np.zeros(1)
        axes[0].scatter(xs, i + jitter, color=COLORS[model], s=22, marker=MARKERS[model])
        axes[0].scatter([float(np.mean(xs))], [i], color="#222222", marker="|", s=95)
    axes[0].set_yticks(range(4), [DISPLAY[model] for model in models])
    axes[0].set_ylim(3.5, -.5)
    axes[0].set_xlabel("Recorded training time (" + ("min" if unit == 60 else "s") + ")")
    axes[0].set_title("Training\nEach completed model seed", fontsize=7.4)
    for i, method in enumerate(methods):
        mean, low, high = _statistics(_metric_row(rows, method, "seconds"), "seconds")
        if mean <= 0 or mean - low <= 0:
            raise EvidenceUnavailable("Positive recorded prediction times required for the logarithmic display")
        axes[1].errorbar(mean, i, xerr=[[low], [high]], fmt=MARKERS[method], color=COLORS[method],
                         markersize=4, capsize=2, elinewidth=.9)
    axes[1].set_xscale("log")
    # Plain numeric strings avoid mathtext's 0.7x superscript glyphs below5pt.
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, position: format(value, ".3g")))
    axes[1].set_yticks(range(len(methods)), [DISPLAY[method] for method in methods])
    axes[1].set_ylim(len(methods) - .5, -.5)
    axes[1].set_xlabel("Prediction time (s per case)", fontsize=7)
    axes[1].set_title("Case prediction\n95% parent-bootstrap intervals", fontsize=7.4)
    for ax in axes:
        ax.grid(axis="x", color="#EAEAEA", linewidth=.6)
    _panel_labels(axes)
    fig.text(.5, .095, "Shared-hardware wall-clock records; not an isolated speed benchmark.",
             ha="center", fontsize=6.6)
    fig.text(.5, .025, "Training: seed totals and mean ticks. Prediction: original durations,\n"
             "not cache lookup latency.", ha="center", fontsize=6.5, linespacing=1.25)
    export(fig, out, "runtime", {
        "question": "What compute costs were actually recorded?", "claim": "Training and case-level prediction costs have separate measurement units and limits.",
        "archetype": "quantitative grid", "panels": {"a": "Completed-epoch training sums by seed", "b": "Recorded prediction duration"},
        "training_interval": "No inferential interval; all model-seed totals are shown with their mean.",
        "prediction_interval": meta.get("bootstrap"),
        "limits": "Training sums exclude preprocessing, unlogged interruptions and waits. Shared GPU/load conditions preclude isolated speed claims. Cached durations refer to their original computation and are not new repetitions."},
        sources, alignment_options=_alignment(axes, 2), draft=draft)


def retention(run, out, manifest, draft=False):
    _guard_public_output(run, out, manifest, draft)
    folder = run / "revision_robustness"
    status = _read(folder / "status.json")
    if status.get("status") not in ("complete", "complete_subset_diagnostic") and not (draft and status.get("status") == "draft_smoke"):
        raise EvidenceUnavailable("Retention diagnostics are not complete")
    source = folder / "retention_summary.csv"
    rows = _rows(source)
    methods = [method for method in DISPLAY if method in {row["method"] for row in rows}]
    if not methods:
        raise EvidenceUnavailable("No registered retention methods")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 85 / 25.4))
    fig.subplots_adjust(left=.115, right=.98, top=.84, bottom=.43, wspace=.39)
    for ax, metric, ylabel in zip(axes, ("blind_iou", "observed_error"),
                                  ("Blind-region IoU", "Observed label error")):
        for method in methods:
            selected = sorted([row for row in rows if row["method"] == method], key=lambda row: int(row["holes"]))
            if [int(row["holes"]) for row in selected] != [2, 4, 6, 8, 10]:
                raise EvidenceUnavailable("Complete nested retention conditions are required for each method")
            xs = np.asarray([float(row["retention_fraction"]) * 100 for row in selected])
            estimates = [_statistics(row, metric) for row in selected]
            mean, low, high = np.asarray(estimates).T
            ax.errorbar(xs, mean, yerr=[low, high], color=COLORS[method], marker=MARKERS[method],
                        markersize=3, linewidth=.9, elinewidth=.6, capsize=1.7, label=DISPLAY[method])
        ax.set(xlabel="Retained holes (%)", ylabel=ylabel, xlim=(16, 104))
        ax.set_xticks([20, 40, 60, 80, 100])
        ax.grid(color="#ECECEC", linewidth=.5)
    _panel_labels(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52, .085), ncol=3,
               fontsize=6.5, frameon=False, columnspacing=1., handlelength=1.5)
    parents = status.get("independent_test_parents", status.get("case_count"))
    fig.text(.5, .025, f"Fixed {parents} synthetic parents; nested ten-hole budget.\n"
             "Equal-parent means and 95% parent-bootstrap intervals.",
             ha="center", fontsize=6.3, linespacing=1.25)
    export(fig, out, "retention", {
        "question": "How does reconstruction change when the same observation budget is retained?",
        "claim": "Nested observations provide a matched retention comparison without asserting a fitted deterioration law.",
        "archetype": "quantitative grid", "panels": {"a": "Blind geometry", "b": "Observed consistency"},
        "methods": methods, "conditions": "2/4/6/8/10 nested holes, mask seed42, no label noise",
        "independent_test_parents": parents, "diagnostic_scope": "Registered subset; not the full primary test cohort",
        "replicate_unit": "Generating parent; model, case and mask repetitions are not conflated."},
        [manifest, source, folder / "status.json", folder / "selection.json"],
        alignment_options=_alignment(axes, 2), draft=draft)


def _diagnostic_case(run, manifest, protocol, draft):
    folder = run / "uncertainty"
    identity = _read(folder / "identity.json")
    if identity.get("manifest_sha256") != _hash(manifest):
        raise ValueError("Local diagnostic and synthetic manifest identities differ")
    if identity.get("smoke") and not draft:
        raise EvidenceUnavailable("Smoke diagnostics require the explicit draft plotting mode")
    selection = _read(folder / "selection.json")
    ids = selection.get("test", [])
    if not ids or ids != identity.get("selection", {}).get("test"):
        raise ValueError("Registered diagnostic selection is absent or inconsistent")
    if not draft and (len(ids) != 30 or identity.get("settings", {}).get("candidate_count") != 100):
        raise EvidenceUnavailable("Formal local-variability figure requires the registered30-parent, K100 diagnostic")
    entries = {row["sample_id"]: row for row in protocol["samples"] if row["split"] == "test"}
    if any(sample not in entries for sample in ids):
        raise ValueError("Diagnostic selection contains a case outside the synthetic test split")
    sample = ids[0]  # The first ID in the predeclared hash ordering, never a score-selected case.
    case_dir = folder / "test" / "cases" / sample / "joint" / "init0"
    complete = _read(case_dir / "complete.json")
    if complete.get("status") != "complete":
        raise EvidenceUnavailable("The preselected primary diagnostic case is incomplete")
    path = case_dir / "adaptation.npz"
    _checked_artifact(path, complete)
    with np.load(path, allow_pickle=False) as data:
        truth, observation = data["target"].copy(), data["observation"].copy()
    entry = entries[sample]
    grid = (manifest.parent / entry["path"]).resolve()
    if not grid.is_relative_to(manifest.parent.resolve()) or _hash(grid) != entry["sha256"]:
        raise ValueError("Selected synthetic target file is outside or differs from its manifest")
    if not np.array_equal(np.load(grid, allow_pickle=False), truth):
        raise ValueError("Saved diagnostic target and manifest target differ")
    if (truth.ndim != 3 or observation.shape != (2, *truth.shape)
            or not np.isfinite(observation).all() or not np.isin(observation[1], [0, 1]).all()):
        raise ValueError("Invalid saved synthetic observation tensor")
    known = observation[1].astype(bool)
    if not known.any() or not np.array_equal(observation[0][known], truth[known]):
        raise ValueError("Saved primary observation labels differ from the synthetic reference")
    return {"sample": sample, "folder": folder, "case_dir": case_dir, "complete": complete,
            "identity": identity, "truth": truth, "observation": observation,
            "sources": [manifest, folder / "identity.json", folder / "selection.json", path, grid,
                        case_dir / "complete.json"]}


def _checked_artifact(path, complete):
    if not Path(path).is_file():
        raise EvidenceUnavailable("Required saved diagnostic artifact is absent: " + Path(path).name)
    expected = complete.get("artifacts", {}).get(Path(path).name)
    if not expected or _hash(path) != expected:
        raise ValueError("Saved diagnostic artifact checksum differs: " + Path(path).name)


def _plane(observation):
    """A true vertical plane through two actual observed XY columns."""
    mask, labels = observation[1].astype(bool), observation[0]
    columns = sorted(tuple(int(value) for value in row) for row in np.argwhere(mask.any(axis=0)))
    if len(columns) < 2:
        raise EvidenceUnavailable("A vertical section requires two distinct observed columns")
    positive = [column for column in columns if np.any(labels[:, column[0], column[1]][mask[:, column[0], column[1]]] > .5)]
    first = positive[0] if positive else columns[0]
    second = min((column for column in columns if column != first),
                 key=lambda column: (-sum((a - b) ** 2 for a, b in zip(column, first)), column))
    start = np.asarray(first, dtype=float)
    direction = np.asarray(second, dtype=float) - start
    separation = float(np.linalg.norm(direction))
    direction /= separation
    lower, upper = -float("inf"), float("inf")
    for center, delta, length in zip(start, direction, mask.shape[1:]):
        if abs(delta) > 1e-12:
            endpoints = sorted([(-.5 - center) / delta, (length - .5 - center) / delta])
            lower, upper = max(lower, endpoints[0]), min(upper, endpoints[1])
    # Sample pixel centers, not display boundaries: imshow's extent then agrees
    # exactly with the coordinates used for interpolation and contouring.
    sample_count = max(2, int(math.ceil(upper - lower)))
    spacing = (upper - lower) / sample_count
    points = lower + (np.arange(sample_count) + .5) * spacing
    xy = start[:, None] + direction[:, None] * points[None, :]
    known_z, known_y, known_x = np.where(mask)
    offsets = np.stack([known_y, known_x], axis=1) - start
    projected = offsets @ direction
    perpendicular = offsets - projected[:, None] * direction
    on_plane = np.linalg.norm(perpendicular, axis=1) < 1e-7
    return {"first_column_yx": list(first), "second_column_yx": list(second),
            "first_column_rule": "First lexicographic observed occupied column, else first observed column",
            "second_column_rule": "Farthest observed column from first; lexicographic tie break",
            "xy": xy, "u": points, "extent": [float(lower), float(upper), -.5, mask.shape[0] - .5],
            "horizontal_samples": sample_count, "horizontal_spacing_voxels": float(spacing),
            "horizontal_centers": "extent_min + (index + 0.5) * spacing",
            "hole_u": [0., separation], "sample_u": projected[on_plane], "sample_z": known_z[on_plane],
            "sample_labels": labels[known_z[on_plane], known_y[on_plane], known_x[on_plane]],
            "shape": list(mask.shape), "interpolation": "Nearest voxel center; identical plane and extent for every panel"}


def _section(volume, plane):
    from scipy.ndimage import map_coordinates
    z = np.arange(volume.shape[0], dtype=float)[:, None]
    coordinates = np.broadcast_arrays(z, plane["xy"][0][None, :], plane["xy"][1][None, :])
    return map_coordinates(np.asarray(volume, dtype=float), np.asarray(coordinates),
                           order=0, mode="nearest", prefilter=False)


def _plane_panel(ax, plane, truth_section, title, image=None, *, cmap="Blues", vmin=0, vmax=1, contour=True, observations=True):
    artist = None
    if image is not None:
        artist = ax.imshow(image, origin="lower", extent=plane["extent"], interpolation="nearest",
                           cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        if contour and np.min(truth_section) < .5 < np.max(truth_section):
            ax.contour(plane["u"], np.arange(truth_section.shape[0]), truth_section,
                       levels=[.5], colors="#222222", linewidths=.55, linestyles="--")
    if observations:
        for u in plane["hole_u"]:
            ax.axvline(u, color="#777777", linewidth=.4, linestyle=":", zorder=3)
        for occupied in (False, True):
            selected = (plane["sample_labels"] >= .5) == occupied
            ax.scatter(plane["sample_u"][selected], plane["sample_z"][selected], s=9,
                       facecolor="#AA4466" if occupied else "#FFFFFF", edgecolor="#333333", linewidth=.35, zorder=5)
    ax.set_xlim(plane["extent"][:2]); ax.set_ylim(plane["extent"][2:])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=6.8, pad=5)
    ax.tick_params(length=2, labelsize=6)
    ax.set_xticks([0, plane["hole_u"][1]])
    ax.set_xticklabels(["0", format(plane["hole_u"][1], ".0f")])
    ax.set_yticks([0, plane["shape"][0] - 1])
    return artist


def _plane_record(plane):
    return {key: value for key, value in plane.items()
            if key not in ("xy", "u", "sample_u", "sample_z", "sample_labels")}


def drillhole_sections(run, out, manifest, protocol, draft=False):
    _guard_public_output(run, out, manifest, draft)
    case = _diagnostic_case(run, manifest, protocol, draft)
    folder = run / "test" / "seed11"
    meta, intent = _read(folder / "evaluation.json"), _read(folder / "evaluation_intent.json")
    if not meta.get("complete") or meta.get("manifest_sha256") != _hash(manifest):
        raise EvidenceUnavailable("Complete identity-matched seed11 test evaluation is required")
    rows = [row for row in _rows(folder / "per_case.csv") if row["sample_id"] == case["sample"]]
    methods = _methods(rows, draft)
    if (case["identity"]["settings"]["mask_seed"] != meta["mask_seed"]
            or int(np.count_nonzero(case["observation"][1])) != int(intent["num_holes"]) * int(intent["samples_per_hole"])):
        raise ValueError("Main evaluation and saved diagnostic observation masks differ")
    plane = _plane(case["observation"])
    truth_section = _section(case["truth"], plane)
    panels = [("Reference\noccupancy", truth_section), ("Observed\nlabels only", None)]
    sources = case["sources"] + [folder / "evaluation.json", folder / "evaluation_intent.json", folder / "per_case.csv"]
    thresholds = {}
    for method in methods:
        matches = [row for row in rows if row["method"] == method]
        if len(matches) != 1:
            raise ValueError("Duplicate or missing selected-case method prediction")
        row = matches[0]
        path = (folder / row["prediction_file"]).resolve()
        if not path.is_relative_to(folder.resolve()) or _hash(path) != row["prediction_sha256"]:
            raise ValueError("Synthetic prediction path/hash differs")
        probability = np.load(path, allow_pickle=False)
        if (probability.shape != case["truth"].shape or not np.isfinite(probability).all()
                or np.min(probability) < 0 or np.max(probability) > 1):
            raise ValueError("Prediction and reference grids differ")
        threshold = float(row["threshold_used"])
        thresholds[method] = threshold
        title = DISPLAY[method].replace("GeoVAE + ", "GeoVAE\n+ ")
        title = title.replace("Indicator kriging", "Indicator\nkriging").replace("MPS (GENESIM)", "MPS\n(GENESIM)")
        panels.append((title, _section((probability >= threshold).astype(float), plane)))
        sources.append(path)
    ncols, nrows = 4, math.ceil(len(panels) / 4)
    fig, grid = plt.subplots(nrows, ncols, figsize=(WIDTH, 90 / 25.4), squeeze=False)
    fig.subplots_adjust(left=.065, right=.98, top=.87, bottom=.18, hspace=.67, wspace=.48)
    all_axes = list(grid.ravel()); axes = all_axes[:len(panels)]
    unused_axes = all_axes[len(panels):]
    for ax in unused_axes:
        ax.axis("off")
    for ax, (title, image) in zip(axes, panels):
        _plane_panel(ax, plane, truth_section, title, image)
    for ax in axes[:ncols * (nrows - 1)]:
        ax.tick_params(labelbottom=False)
    _panel_labels(axes)
    fig.text(.5, .080, "Same plane. Dashed outline: reference. Filled/open circles: occupied/barren.\n"
             "Horizontal: voxel distance from first column. Vertical: z index.",
             ha="center", fontsize=6.2, linespacing=1.2)
    fig.text(.5, .025, "Fixed first hash-selected case: " + case["sample"], ha="center", fontsize=6.2)
    export(fig, out, "drillhole_sections", {
        "question": "How do all methods reconstruct the same plane through actual observed columns?",
        "claim": "A prespecified synthetic case is shown under identical sampling coordinates, rather than favorable or unrelated views.",
        "archetype": "image plate + quant", "case_selection": "First case in registered hash-ordered test selection",
        "sample_id": case["sample"], "plane": _plane_record(plane), "thresholds": thresholds,
        "reference_occupied_plane_samples": int(np.count_nonzero(truth_section)),
        "methods": methods, "display": "Binary occupancy at each method's frozen validation threshold; no interpolation of borehole trajectories."},
        sources, alignment_options=_alignment(axes, ncols, exclude=unused_axes), draft=draft)


def local_variability(run, out, manifest, protocol, draft=False):
    _guard_public_output(run, out, manifest, draft)
    case = _diagnostic_case(run, manifest, protocol, draft)
    folder, complete = case["case_dir"], case["complete"]
    source = case["folder"] / "test" / "per_case_acceptance.csv"
    matching = [row for row in _rows(source) if row["sample_id"] == case["sample"] and row["mode"] == "joint"
                and int(row["initialization"]) == 0 and np.isclose(float(row["variance"]), .01)
                and np.isclose(float(row["threshold"]), .001)]
    if len(matching) != 1:
        raise EvidenceUnavailable("The fixed variance.01 / tau.001 / joint / init0 diagnostic is incomplete")
    row = matching[0]
    probability_path = folder / "candidates_var0.01.npz"
    candidate_path = folder / "candidates_var0.01.csv"
    _checked_artifact(probability_path, complete); _checked_artifact(candidate_path, complete)
    with np.load(probability_path, allow_pickle=False) as data:
        probabilities = data["probabilities"].copy()
    if probabilities.shape != (int(row["candidate_count"]), *case["truth"].shape):
        raise ValueError("Saved candidate count/grid shape differs")
    if not np.isfinite(probabilities).all() or np.min(probabilities) < 0 or np.max(probabilities) > 1:
        raise ValueError("Saved candidate probabilities are invalid")
    candidates = _rows(candidate_path)
    accepted = sorted(int(candidate["candidate_index"]) for candidate in candidates
                      if float(candidate["observed_mse"]) <= .001 and int(candidate["observed_violation_count"]) == 0)
    if len(accepted) != int(row["accepted_count"]) or accepted != json.loads(row["accepted_indices"]):
        raise ValueError("Accepted candidate indices differ from the stored screening criterion")
    mean, variance = None, None
    sources = case["sources"] + [source, probability_path, candidate_path]
    if accepted:
        path = folder / "mean_var0.01_tau0.001.npy"
        _checked_artifact(path, complete)
        mean = np.load(path, allow_pickle=False)
        if not np.allclose(mean, probabilities[accepted].mean(axis=0, dtype=np.float64), rtol=2e-6, atol=2e-7):
            raise ValueError("Saved accepted mean differs from candidate probabilities")
        sources.append(path)
    if len(accepted) >= 2:
        path = folder / "variance_var0.01_tau0.001.npy"
        _checked_artifact(path, complete)
        variance = np.load(path, allow_pickle=False)
        if not np.allclose(variance, probabilities[accepted].var(axis=0, ddof=1, dtype=np.float64), rtol=2e-6, atol=2e-7):
            raise ValueError("Saved sample variance differs from accepted candidates")
        sources.append(path)
    # Preserve accepted-index order. Do not select on geometry scores or on how
    # visibly different a candidate looks; repeated geometries remain apparent.
    displayed = accepted[:3]
    signatures = {hashlib.sha256(np.packbits(probabilities[index] >= .5).tobytes()).hexdigest() for index in accepted}
    plane = _plane(case["observation"])
    truth_section = _section(case["truth"], plane)
    images = [truth_section, _section(mean, plane) if mean is not None else None,
              _section(variance, plane) if variance is not None else None]
    images += [_section((probabilities[index] >= .5).astype(float), plane) for index in displayed]
    images += [None] * (6 - len(images))
    titles = ["Reference occupancy", "Accepted probability\nmean", "Accepted sample\nvariance"]
    titles += ["Accepted candidate " + str(index) for index in displayed]
    titles += ["No further accepted\ncandidate" if accepted else "No accepted candidate"] * (6 - len(titles))
    fig, grid = plt.subplots(2, 3, figsize=(WIDTH, 3.9))
    fig.subplots_adjust(left=.065, right=.96, top=.86, bottom=.30, hspace=.58, wspace=.27)
    axes = list(grid.ravel()); mean_artist = variance_artist = None
    vmax = float(np.max(variance)) if variance is not None else None
    for index, (ax, title, image) in enumerate(zip(axes, titles, images)):
        artist = _plane_panel(ax, plane, truth_section, title, image,
                              cmap="magma" if index == 2 else "Blues", vmax=max(vmax, 1e-12) if index == 2 and vmax is not None else 1,
                              contour=index != 2, observations=image is not None)
        if index == 1:
            mean_artist = artist
        if index == 2:
            variance_artist = artist
        if image is None:
            reason = ("S = 0\nMean undefined" if index == 1 else
                      "S < 2\nSample variance\nundefined" if index == 2 else "Unavailable")
            ax.text(.5, .5, reason, transform=ax.transAxes, ha="center", va="center", fontsize=7, color=GRAY)
    _panel_labels(axes)
    colorbars = []
    if mean_artist is not None:
        cax = fig.add_axes([.24, .16, .20, .018]); colorbars.append(cax)
        fig.colorbar(mean_artist, cax=cax, orientation="horizontal", ticks=[0, .5, 1]).set_label("Mean occupancy probability", fontsize=6.5)
        cax.xaxis.set_label_position("top")
        cax.tick_params(labelsize=6)
    if variance_artist is not None:
        cax = fig.add_axes([.62, .16, .20, .018]); colorbars.append(cax)
        cb = fig.colorbar(variance_artist, cax=cax, orientation="horizontal")
        cb.set_label("Sample variance (ddof = 1)", fontsize=6.5)
        cax.xaxis.set_label_position("top")
        if vmax == 0:
            cb.set_ticks([0])
        cax.tick_params(labelsize=6)
    fig.text(.5, .080, "First hash-selected case; joint, init 0, variance 0.01, tau 0.001.",
             ha="center", fontsize=6.6)
    fig.text(.5, .025, f"Accepted S = {len(accepted)}/{len(probabilities)}; distinct binary geometries = {len(signatures)}.\n"
             "Candidate indices follow acceptance order.", ha="center", fontsize=6.5, linespacing=1.25)
    export(fig, out, "local_variability", {
        "question": "What local output variation remains after observation screening, and when is it undefined?",
        "claim": "The accepted mean and sample variance describe this local proposal experiment, not a validated conditional posterior.",
        "archetype": "image plate + quant", "sample_id": case["sample"], "mode": "joint", "initialization": 0,
        "diagnostic_test_parents": len(case["identity"]["selection"]["test"]), "displayed_cases": 1,
        "variance": .01, "threshold": .001, "candidate_count": len(probabilities), "accepted_count": len(accepted),
        "mean_defined": bool(accepted), "sample_variance_defined": len(accepted) >= 2,
        "displayed_candidate_indices": displayed, "distinct_accepted_binary_geometries": len(signatures),
        "candidate_selection": "First three accepted indices; no outcome/visual-diversity selection. Binary duplicates are not hidden.",
        "plane": _plane_record(plane), "undefined_policy": "S0: no mean or variance. S1: mean only. No optimized-prediction or zero-array substitution."},
        sources, alignment_options=_alignment(axes, 3, exclude=colorbars), draft=draft)


def accepted_rate(run, out, manifest, draft=False):
    _guard_public_output(run, out, manifest, draft)
    folder = run / "uncertainty"
    identity, status = _read(folder / "identity.json"), _read(folder / "test" / "status.json")
    if identity.get("manifest_sha256") != _hash(manifest) or status.get("status") != "complete":
        raise EvidenceUnavailable("Complete identity-matched synthetic acceptance diagnostics are required")
    source = folder / "test" / "per_case_acceptance.csv"
    rows = _rows(source)
    modes = [mode for mode in ("latent", "joint") if mode in {row["mode"] for row in rows}]
    if set(modes) != set(identity["settings"]["modes"]):
        raise ValueError("Acceptance modes differ from the registered diagnostic")
    expected = len(identity["selection"]["test"]) * len(identity["settings"]["initializations"])
    variances = sorted(float(x) for x in identity["settings"]["variances"])
    thresholds = sorted(float(x) for x in identity["settings"]["thresholds"])
    combinations = [(variance, tau) for variance in variances for tau in thresholds]
    fig, grid = plt.subplots(len(modes), 2, figsize=(WIDTH, 100 / 25.4), squeeze=False)
    height_pt = fig.get_figheight() * 72
    fig.subplots_adjust(left=.235, right=.98, top=.83, bottom=.25 if len(modes) == 1 else .19,
                        hspace=.80, wspace=.31)
    axes = list(grid.ravel()); parent_records, coverage_records, strata_records = [], [], []
    label_composition_available = all(row.get("observed_label_composition") for row in rows)
    for mode_index, mode in enumerate(modes):
        left, right = grid[mode_index]
        for group_index, (variance, tau) in enumerate(combinations):
            selected = [row for row in rows if row["mode"] == mode
                        and np.isclose(float(row["variance"]), variance) and np.isclose(float(row["threshold"]), tau)]
            if len(selected) != expected:
                raise EvidenceUnavailable("Acceptance condition has incomplete case/start coverage")
            by_parent = defaultdict(list)
            for row in selected:
                count, total = int(row["accepted_count"]), int(row["candidate_count"])
                if total < 1 or not 0 <= count <= total or not np.isclose(float(row["acceptance_rate"]), count / total):
                    raise ValueError("Invalid acceptance count/rate")
                by_parent[row["parent_id"]].append(count / total)
            values = [float(np.mean(by_parent[parent])) for parent in sorted(by_parent)]
            jitter = np.linspace(-.18, .18, len(values)) if len(values) > 1 else np.zeros(1)
            left.scatter(values, group_index + jitter, s=10, color=BLUE, alpha=.5, linewidths=0)
            left.scatter([np.mean(values)], [group_index], marker="|", s=90, color="#222222")
            for parent, value in zip(sorted(by_parent), values):
                parent_records.append({"mode": mode, "variance": variance, "threshold": tau,
                                       "parent_id": parent, "mean_acceptance_over_starts": value,
                                       "starts": len(by_parent[parent])})
            counts = [sum(int(row["accepted_count"]) == 0 for row in selected),
                      sum(int(row["accepted_count"]) == 1 for row in selected),
                      sum(int(row["accepted_count"]) >= 2 for row in selected)]
            offset = 0.
            for n, color, label in zip(counts, ("#BBBBBB", "#DDAA66", BLUE), ("S = 0", "S = 1", "S >= 2")):
                fraction = n / len(selected)
                right.barh(group_index, fraction, left=offset, height=.62, color=color,
                           label=label if group_index == 0 else None)
                offset += fraction
            coverage_records.append({"mode": mode, "variance": variance, "threshold": tau,
                                     "case_starts": len(selected), "zero_accepted": counts[0],
                                     "one_accepted": counts[1], "variance_defined": counts[2]})
            if label_composition_available:
                for composition in ("all_barren", "all_occupied", "mixed"):
                    subgroup = [row for row in selected if row["observed_label_composition"] == composition]
                    if subgroup:
                        strata_records.append({"mode": mode, "variance": variance, "threshold": tau,
                                               "observed_label_composition": composition,
                                               "case_starts": len(subgroup),
                                               "independent_parents": len({row["parent_id"] for row in subgroup}),
                                               "zero_accepted": sum(int(row["accepted_count"]) == 0 for row in subgroup),
                                               "one_accepted": sum(int(row["accepted_count"]) == 1 for row in subgroup),
                                               "variance_defined": sum(int(row["accepted_count"]) >= 2 for row in subgroup),
                                               "mean_acceptance_rate": float(np.mean([float(row["acceptance_rate"]) for row in subgroup]))})
        for ax in (left, right):
            ax.set_ylim(len(combinations) - .5, -.5); ax.set_xlim(-.025, 1.025)
            ax.set_yticks(range(len(combinations)))
            ax.set_xticks([0, .5, 1])
        left.set_yticklabels([f"v={variance:g}; tau={tau:g}" for variance, tau in combinations], fontsize=6.4)
        right.set_yticklabels([])
        left.set_title(("Latent only" if mode == "latent" else "Latent + LoRA") + ": acceptance", fontsize=7.2)
        right.set_title("Retained ensemble coverage", fontsize=7.2)
        left.set_xlabel("Accepted fraction:\nparent means over starts", fontsize=6.6)
        right.set_xlabel("Fraction of registered\ncase-start runs", fontsize=6.6)
    _panel_labels(axes)
    handles, labels = grid[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.72, 25 / height_pt), ncol=3,
               fontsize=6.5, frameon=False, handlelength=1.3, columnspacing=1.)
    fig.text(.5, 7 / height_pt, f"Parents = {len(identity['selection']['test'])}; starts = {len(identity['settings']['initializations'])} per parent. "
             "S >= 2: sample variance defined.\nEligibility is not posterior calibration.",
             ha="center", fontsize=6.4, linespacing=1.25)
    derived = [_source_table(out, "acceptance_parent_distribution", parent_records),
               _source_table(out, "acceptance_coverage", coverage_records)]
    if strata_records:
        derived.append(_source_table(out, "acceptance_by_observed_labels", strata_records))
    export(fig, out, "accepted_rate", {
        "question": "How sensitive are acceptance and eligible-ensemble coverage to screening and perturbation scale?",
        "claim": "All predeclared variance/threshold conditions and zero/one-acceptance outcomes remain visible.",
        "archetype": "quantitative grid", "replicate_unit": "Generating parent; starts are averaged within parent for acceptance points.",
        "coverage_unit": "Recorded case-start, explicitly distinct from independent parents",
        "interval": "Raw parent means and their mean ticks; no confidence interval is claimed.",
        "modes": modes, "case_starts_per_condition": expected,
        "independent_test_parents": len(identity["selection"]["test"]), "model_seed": identity["settings"]["seed"],
        "observed_label_strata": "Archived separately" if label_composition_available else "Unavailable in this legacy draft table; no composition was invented",
        "exclusions": "None; all registered variances, thresholds, cases and initializations are required."},
        [manifest, source, folder / "identity.json", folder / "test" / "status.json", *derived],
        alignment_options=_alignment(axes, 2), draft=draft)


def quantitative(run, out, *, draft=False, manifest_path=None):
    run = Path(run)
    Path(out).mkdir(parents=True, exist_ok=True)
    if (run / "smoke_identity.json").is_file() and not draft:
        raise ValueError("CPU smoke outputs require --draft; they cannot silently become paper evidence")
    manifest, protocol = _synthetic_manifest(run, manifest_path)
    _guard_public_output(run, out, manifest, draft)
    generated, pending = [], []
    status_path = Path(out) / "quantitative_figures.json"
    _write_json(status_path, {"status": "rendering", "script_sha256": _hash(__file__),
                              "manifest_sha256": _hash(manifest), "draft": draft})
    functions = (main_comparison, calibration, runtime, retention, drillhole_sections, local_variability, accepted_rate)
    for function in functions:
        try:
            if function in (drillhole_sections, local_variability):
                function(run, out, manifest, protocol, draft)
            else:
                function(run, out, manifest, draft)
            generated.append(function.__name__)
            print("[FIGURES] Saved " + function.__name__, flush=True)
        except EvidenceUnavailable as error:
            pending.append({"figure": function.__name__, "reason": str(error)})
            print("[FIGURES] Pending " + function.__name__ + ": " + str(error), flush=True)
        except Exception as error:
            _write_json(status_path, {"status": "failed", "failed_figure": function.__name__,
                                      "error_type": type(error).__name__, "generated": generated,
                                      "script_sha256": _hash(__file__), "draft": draft})
            raise
    report = {"status": "draft" if draft else "complete" if not pending else "partial",
              "generated": generated, "pending": pending, "data_policy": "synthetic_only",
              "script_sha256": _hash(__file__), "manifest_sha256": _hash(manifest)}
    _write_json(status_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run")
    parser.add_argument("--manifest", help="Explicit independent synthetic protocol manifest, if not archived inside --run")
    parser.add_argument("--draft", action="store_true", help="Explicitly watermark CPU smoke renders; never use them as manuscript evidence")
    parser.add_argument("--schematics", action="store_true")
    args = parser.parse_args(argv)
    if args.run and not args.schematics:
        manifest, _ = _synthetic_manifest(Path(args.run), args.manifest)
        _guard_public_output(Path(args.run), args.out, manifest, args.draft)
    schematics(args.out)
    if args.run and not args.schematics:
        quantitative(args.run, args.out, draft=args.draft, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
