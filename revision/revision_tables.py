"""Generate manuscript table bodies from verified new revision results only.

No prose conclusion or best-method selection is inferred by this formatter.
Undefined values remain missing. Private real geometry is never read.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

METHODS = ("idw", "rbf", "ik", "mps", "unet", "base_vae", "cgan",
           "geovae", "geovae_latent", "geovae_joint")
LABEL = {"idw": "IDW", "rbf": "RBF", "ik": "Indicator kriging",
         "mps": "MPS (GENESIM)", "unet": "3D U-Net", "base_vae": "Basic VAE",
         "cgan": "3D cGAN", "geovae": "GeoVAE", "geovae_latent": "GeoVAE + z",
         "geovae_joint": "GeoVAE + z + LoRA"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _number(value):
    if value in (None, "", "None", "nan"):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def fmt(value, digits=3):
    value = _number(value)
    return "--" if value is None else f"{value:.{digits}f}"


def interval(row, metric, digits=3):
    mean, lo, hi = (row.get(metric + suffix) for suffix in ("_mean", "_ci_low", "_ci_high"))
    if _number(mean) is None:
        return "--"
    return fmt(mean, digits) + " [" + fmt(lo, digits) + ", " + fmt(hi, digits) + "]"


def table(headers, rows):
    lines = [r"\small\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "r" * (len(headers)-1) + r"}\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines.append(r"\bottomrule\end{tabular}")
    return "\n".join(lines) + "\n"


def _accepted_tables(run):
    rows = _rows(run / "uncertainty/test/per_case_acceptance.csv")
    if len({r["sample_id"] for r in rows}) != 30:
        raise RuntimeError("Acceptance table requires all 30 preselected test cases")
    groups = defaultdict(list)
    for row in rows:
        groups[(float(row["variance"]), float(row["threshold"]))].append(row)
    if len(groups) != 9 or any(len(group) != 90 for group in groups.values()):
        raise RuntimeError("Acceptance sensitivity is incomplete")
    formatted, evidence = [], []
    for (variance, tau), group in sorted(groups.items()):
        counts = [int(r["accepted_count"]) for r in group]
        proposals = sum(int(r["candidate_count"]) for r in group)
        if proposals != 9000:
            raise RuntimeError("Formal candidate count differs from 100 per initialization")
        record = {"variance": variance, "threshold": tau,
                  "test_parents": 30, "case_initializations": 90,
                  "candidates": proposals, "accepted": sum(counts),
                  "acceptance_fraction": sum(counts) / proposals,
                  "zero_accepted": sum(c == 0 for c in counts),
                  "single_accepted": sum(c == 1 for c in counts),
                  "variance_defined": sum(c >= 2 for c in counts)}
        evidence.append(record)
        formatted.append([fmt(variance), f"{tau:g}", fmt(record["acceptance_fraction"]),
                          str(record["zero_accepted"]), str(record["single_accepted"]),
                          str(record["variance_defined"])])
    return table(["Proposal variance", r"$\tau$", "Accepted fraction", "$S=0$", "$S=1$", r"$S\geq2$"], formatted), evidence


def _runtime_table(run, main):
    events = [json.loads(line) for line in (run / "stage_events.jsonl").read_text().splitlines()]
    from datetime import datetime
    starts, stage_seconds = {}, defaultdict(float)
    for event in events:
        stage = event["stage"]
        if not (stage.startswith("train/") or
                stage.startswith("ablation/train/base_no_observed/") or
                stage.startswith("ablation/train/attention_no_position/")):
            continue
        moment = datetime.fromisoformat(event["time_utc"].replace("Z", "+00:00"))
        if event["event"] == "RUN": starts[stage] = moment
        if event["event"] in ("DONE", "ERROR") and stage in starts:
            stage_seconds[stage] += (moment-starts.pop(stage)).total_seconds()
    migration_path = run / "training_migration.json"
    if migration_path.is_file():
        migration = _json(migration_path)
        if migration.get("status") != "verified_same_training_recipe_migration":
            raise RuntimeError("Training time migration is unverified")
        for model in migration["models"]:
            stage_seconds[model["stage"]] += float(model["prior_training_wall_seconds"])
    duration = defaultdict(list)
    for stage, seconds in sorted(stage_seconds.items()):
        parts = stage.split("/")
        key = parts[1] if parts[0] == "train" else "ablation:" + parts[2]
        duration[key].append(seconds)
    result = []
    for method in METHODS:
        training = duration.get(method, [])
        train_hours = sum(training) / 3600 if training else None
        result.append([LABEL[method], fmt(train_hours, 2), str(len(training)),
                       fmt(main[method].get("seconds_mean"), 3)])
    for arm, label in (("base_no_observed", "VAE, no obs. loss"),
                       ("attention_no_position", "Attention, no position")):
        training = duration.get("ablation:" + arm, [])
        result.append([label, fmt(sum(training) / 3600 if training else None, 2),
                       str(len(training)), "--"])
    diagnostic = _rows(run / "uncertainty/test/per_case_acceptance.csv")
    optimization, sampling = {}, {}
    for row in diagnostic:
        key = (row["sample_id"], row["mode"], int(row["initialization"]))
        optimization[key] = float(row["optimization_seconds"])
        sampling[(*key, float(row["variance"]))] = float(row["sampling_seconds"])
    if optimization:
        result.append(["Diagnostic adaptation", "--", "--",
                       fmt(sum(optimization.values()) / len(optimization))])
    if sampling:
        result.append(["Local candidates (K=100)", "--", "--",
                       fmt(sum(sampling.values()) / len(sampling))])
    return table(["Method", "Total training (h)", "Training fits", "Prediction (s/case)"], result), dict(duration)


def generate(run_root, output):
    run, output = Path(run_root).resolve(), Path(output).resolve()
    identity = _json(run / "run_identity.json")
    analysis = _json(run / "revision_analysis/analysis.json")
    if identity.get("mode") != "formal" or identity.get("config", "").endswith("smoke16.yaml"):
        raise RuntimeError("Draft runs cannot fill the manuscript")
    if analysis.get("status") != "complete_primary_reanalysis":
        raise RuntimeError("Verified complete primary reanalysis is required")
    if analysis["manifest_sha256"] != identity["manifest_sha256"]:
        raise RuntimeError("Primary evidence belongs to another protocol")
    integrity_path = run / "final_integrity/report.json"
    integrity = _json(integrity_path)
    if integrity.get("status") != "passed" or integrity.get("manifest_sha256") != identity["manifest_sha256"]:
        raise RuntimeError("Final artifact and candidate numerical audit must pass first")
    if analysis.get("analysis_script_sha256") != _sha(Path(__file__).with_name("revision_analysis.py")):
        raise RuntimeError("Reanalysis must be rerun with the final analysis helper")
    config = _json(run / "tuning/selected_config.json")
    if int(config["latent_dim"]) != 256:
        raise RuntimeError("Historical 128-dimensional results are not eligible")
    required = ["revision_analysis/summary.csv", "uncertainty/test/status.json",
                "revision_robustness/status.json", "real_aggregate/real_summary.json",
                "ablation/summaries/test/ablation_summary.csv"]
    for relative in required:
        if not (run / relative).is_file(): raise FileNotFoundError(relative)
    if _json(run / "uncertainty/test/status.json").get("case_count") != 30:
        raise RuntimeError("Incomplete uncertainty diagnostic")
    robust_status = _json(run / "revision_robustness/status.json")
    if robust_status.get("status") != "complete_subset_diagnostic" or robust_status.get("case_count") != 30:
        raise RuntimeError("Incomplete mandatory robustness diagnostic")
    real = _json(run / "real_aggregate/real_summary.json")
    if real.get("status") not in ("complete", "complete_with_exclusions") or real.get("runtime_failures"):
        raise RuntimeError("Retrospective evidence contains unresolved failures")
    main = {r["method"]: r for r in _rows(run / "revision_analysis/summary.csv")}
    if set(main) != set(METHODS): raise RuntimeError("Main method coverage is incomplete")
    primary_rows = [[LABEL[m], interval(main[m], "blind_iou"), fmt(main[m]["blind_dice_mean"]),
                     fmt(main[m]["hd95_mean"], 2), fmt(main[m]["chamfer_mean"], 2),
                     fmt(main[m]["input_error_0p5_mean"]), fmt(main[m]["brier_blind_mean"])] for m in METHODS]
    files = {"primary_table.tex": table(["Method", "Blind IoU [95\\% CI]", "Dice", "HD95", "CD",
                                        "Obs. err. (0.5)", "Brier"], primary_rows)}
    from revision_analysis import analyse_evaluations
    ablation_rows = []
    for arm, label in (("base_no_observed", "VAE, no observed loss"),
                       ("base_observed", "VAE + observed loss"),
                       ("attention_no_position", "Attention, no position"),
                       ("attention_position", "Relative-position attention")):
        dirs = [run / "ablation/evaluations/test" / arm / f"seed{s}" for s in (11,23,37)]
        methods = tuple(_json(dirs[0] / "evaluation.json")["methods"])
        dest = run / "revision_analysis/ablation" / arm
        analyse_evaluations(run / "frozen_inputs/manifest.json", dirs, dest,
                            config_path=run / "ablation/evaluation_configs" / (arm + ".json"),
                            required_methods=methods)
        for row in _rows(dest / "summary.csv"):
            suffix = {"geovae_latent": " + z", "geovae_joint": " + z + LoRA"}.get(row["method"], "")
            ablation_rows.append([label + suffix, interval(row, "blind_iou"),
                                  fmt(row["input_mse_mean"]), fmt(row["input_error_0p5_mean"]),
                                  fmt(row["hidden_error_mean"])])
    files["ablation_table.tex"] = table(["Arm / inference", "Blind IoU [95\\% CI]", "Obs. MSE",
                                        "Obs. err. (0.5)", "Hidden err."], ablation_rows)
    files["acceptance_table.tex"], acceptance = _accepted_tables(run)
    retention = _rows(run / "revision_robustness/retention_summary.csv")
    retention_rows = []
    for method in METHODS:
        rows = {int(r["holes"]): r for r in retention if r["method"] == method}
        if set(rows) != {2,4,6,8,10}: raise RuntimeError("Incomplete nested retention design")
        retention_rows.append([LABEL[method], *[fmt(rows[h]["blind_iou_mean"]) for h in (2,4,6,8,10)]])
    files["retention_table.tex"] = table(["Method", "20\\%", "40\\%", "60\\%", "80\\%", "100\\%"], retention_rows)
    real_rows = []
    for row in real["metrics"]:
        real_rows.append([LABEL[row["method"]], str(row["evaluated_reference_blocks"]),
                          fmt(row["blind_iou_mean"]), fmt(row["blind_dice_mean"]),
                          fmt(row["hd95_mean"],2), fmt(row["chamfer_mean"],2)])
    if set(r[0] for r in real_rows) != set(LABEL.values()):
        raise RuntimeError("Real aggregate method coverage is incomplete")
    files["real_table.tex"] = table(["Method", "Eligible blocks", "Blind IoU", "Dice", "HD95", "CD"], real_rows)
    files["runtime_table.tex"], training_seconds = _runtime_table(run, main)
    source_paths = sorted(set(required + ["final_integrity/report.json",
                         "revision_analysis/analysis.json", "tuning/selected_config.json",
                         "uncertainty/test/per_case_acceptance.csv", "revision_robustness/retention_summary.csv",
                         "stage_events.jsonl"]))
    evidence = {"status": "verified_result_tables", "script_sha256": _sha(__file__),
                "manifest_sha256": identity["manifest_sha256"], "latent_dim": 256,
                "sources": {p: _sha(run/p) for p in source_paths},
                "primary": list(main.values()), "acceptance": acceptance,
                "real_summary": real, "training_stage_seconds": training_seconds,
                "constraint_error_policy": "Raw observed-label error at common probability 0.5; overlap uses frozen method-specific validation thresholds.",
                "interpretation": "No best-method claim is generated. Parent CI conditional on the three fitted deep models; real cohort is descriptive only."}
    output.mkdir(parents=True, exist_ok=True)
    for name, contents in files.items():
        prefix = "% Generated from the new d256 revision only.\n% Manifest SHA256: " + identity["manifest_sha256"] + "\n"
        (output/name).write_text(prefix+contents, encoding="utf-8")
    (output/"table_evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    return output / "table_evidence.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(generate(args.run, args.out))


if __name__ == "__main__":
    main()
