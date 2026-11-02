"""Gated secondary, post-results cross-arm contrasts; saved scalar reanalysis only.

No training, inference, arrays, checkpoints, parameter selection or public export.
The CLI cannot lower coverage, change contrasts, bootstrap seed or metric family.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import sys

ADMIN = Path(__file__).resolve().parent
OUTPUT_BASE = ADMIN / "secondary_ablation_contrasts_outputs"
MANIFEST_SHA = "48ec0924b76a1cca26715bb77566646dacb5fe9a3252471f8e4690aae8e01030"
CORE_SHA = "2bbb423c62e8ca17cd8045d98cc62fcd123d0a0ade27c6dd2ef72cd52c91d5c3"
SEEDS = (11, 23, 37)
REPLICATES, BOOTSTRAP_SEED = 2000, 20261025
METRICS = ("blind_iou", "brier_blind", "observed_mse_truth", "observed_error", "hidden_error")
CONTRASTS = (
    ("extra_observed_loss", "base_observed", "base_no_observed"),
    ("plain_attention", "attention_no_position", "base_observed"),
    ("relative_position", "attention_position", "attention_no_position"),
    ("combined_attention_and_position", "attention_position", "base_observed"),
)
ARMS = {
    "base_no_observed": ("base_vae",), "base_observed": ("base_vae",),
    "attention_no_position": ("geovae",),
    "attention_position": ("geovae", "geovae_latent", "geovae_joint"),
}
CORE_FILES = ("repro_pipeline.py", "experiment_data.py", "model.py", "repro_cgan.py",
              "repro_baselines.py", "repro_metrics.py", "comparisons/models/unet.py")
NUMERICS = {"cublas_workspace_config": ":4096:8", "cudnn_benchmark": False,
            "cudnn_deterministic": True, "deterministic_algorithms": True,
            "deterministic_warn_only": False, "scope": "evaluation_only_training_recipe_unchanged",
            "version": "esi-evaluation-numerics-v1"}


class Blocked(RuntimeError):
    """Missing or conflicting evidence; never substitute original scalar logs."""


def require(condition, message):
    if not condition:
        raise Blocked(message)


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, "Duplicate JSON key: " + key)
        out[key] = value
    return out


class Evidence:
    """Small JSON/CSV/source reader, recording identities without loading arrays."""
    def __init__(self):
        self.content = {}

    def raw(self, path):
        path = Path(path).resolve()
        require(path.suffix in {".json", ".csv", ".py"}, "Forbidden input type")
        if path not in self.content:
            require(path.is_file(), "Required evidence missing: " + str(path))
            before = path.stat()
            require(before.st_size <= 32 * 1024 * 1024, "Input exceeds bounded size")
            data = path.read_bytes()
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    "Evidence changed while reading")
            self.content[path] = data
        return self.content[path]

    def digest(self, path):
        return hashlib.sha256(self.raw(path)).hexdigest()

    def json(self, path):
        return json.loads(self.raw(path), object_pairs_hook=unique_object,
                          parse_constant=lambda value: require(False, "Nonfinite JSON: " + value))

    def csv(self, path):
        reader = csv.DictReader(io.StringIO(self.raw(path).decode("utf-8-sig")))
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
                "Missing/duplicate CSV header")
        rows = list(reader)
        require(all(None not in row and None not in row.values() for row in rows), "Malformed CSV row")
        return rows

    def unchanged(self):
        require(all(p.is_file() and p.read_bytes() == data for p, data in self.content.items()),
                "Input changed after validation; no results written")


def validate(run, plan_path=ADMIN / "claim_evidence_writeup_plan.json"):
    """Return only verified recomputed scalar rows. Missing gates fail before CSV use."""
    run = Path(run).resolve()
    code = run.parent.parent
    e = Evidence()
    status = e.json(run / "revision_status.json")
    integrity = e.json(run / "final_integrity/report.json")
    require(status.get("status") == "experiments_complete", "Experiments are not complete")
    require(integrity.get("status") == "passed" and integrity.get("formal") is True,
            "Full formal final_integrity has not passed")
    required_checks = {"evaluation_numerical_policy", "frozen_protocol", "raw_sources", "primary_results",
                       "uncertainty_numerical_recomputation", "robustness_numerical_recomputation",
                       "ablation_results", "private_real_numerical_recomputation"}
    require(required_checks <= set(integrity.get("checks", {})), "Final audit has missing mandatory stages")
    require(integrity["checks"]["ablation_results"] == {"arms": 4, "verified_case_method_rows": 13500},
            "Final ablation coverage differs")
    require(not any("skipped" in str(v).lower() for v in integrity["checks"].values()), "Skipped integrity stage")
    identity = e.json(run / "run_identity.json")
    require(identity.get("mode") == "formal" and identity.get("limit") == 0
            and identity.get("seeds") == list(SEEDS), "Run is not full three-seed formal run")
    require(identity.get("manifest_sha256") == integrity.get("manifest_sha256") == MANIFEST_SHA
            and identity.get("source_sha256") == integrity.get("source_sha256") == CORE_SHA,
            "Run/final audit manifest or source differs")
    require(identity.get("evaluation_numerical_policy") == NUMERICS, "Numerical strategy differs")
    require(integrity.get("audit_source_sha256") == e.digest(code / "revision_integrity.py"),
            "Final audit source is stale")
    core = hashlib.sha256()
    for name in CORE_FILES:
        digest = e.digest(code / name)
        require(identity.get("code_files", {}).get(name) == digest, "Frozen core file differs: " + name)
        core.update(name.encode()); core.update(digest.encode("ascii"))
    require(core.hexdigest() == CORE_SHA, "Training/evaluation core digest differs")
    require(identity.get("evaluation_numerics_source_sha256") == e.digest(code / "revision_numerics.py"),
            "Numerical helper differs")
    manifest_path = run / "frozen_inputs/manifest.json"
    require(e.digest(manifest_path) == MANIFEST_SHA, "Frozen manifest bytes differ")
    manifest = e.json(manifest_path)
    require(manifest.get("protocol_version") == "procedural-lens-v1"
            and manifest.get("split_counts") == {"train": 3500, "val": 750, "test": 750}
            and manifest.get("parent_counts") == {"train": 700, "val": 150, "test": 150}
            and manifest.get("grid_size") == [32, 32, 32], "Protocol differs")
    samples = manifest["samples"]
    require(len(samples) == len({x["sample_id"] for x in samples}) == 5000, "Manifest sample duplication/count")
    parents = {x["sample_id"]: x["parent_id"] for x in samples if x["split"] == "test"}
    require(len(parents) == 750 and Counter(Counter(parents.values()).values()) == {5: 150},
            "Expected exactly 150 test parents with five variants")
    plan = e.json(plan_path)["future_cross_arm_CI"]
    expected = [{"name": n, "candidate": a, "reference": b} for n, a, b in CONTRASTS]
    require(plan["contrasts"] == expected and plan["all_metrics"] == list(METRICS)
            and plan["bootstrap"]["replicates"] == REPLICATES
            and plan["bootstrap"]["seed"] == BOOTSTRAP_SEED, "Secondary analysis plan changed")
    selected = e.json(run / "tuning/selected_config.json")
    require(selected.get("latent_dim") == 256 and selected.get("grid_size") == [32, 32, 32]
            and selected.get("lambda_observed", 0) > 0, "Wrong selected model configuration")
    script_sha = e.digest(code / "revision_analysis.py")
    all_arms = {}
    for arm, methods in ARMS.items():
        folder = run / "revision_analysis/ablation" / arm
        analysis = e.json(folder / "analysis.json")
        cfg = e.json(run / "ablation/evaluation_configs" / (arm + ".json"))
        wanted = dict(selected)
        if arm == "base_no_observed": wanted["lambda_observed"] = 0.0
        if arm.startswith("attention_"):
            wanted.update(use_transformer=True, use_relative_position=arm == "attention_position")
        require(cfg == wanted, "Unexpected cross-arm config change: " + arm)
        th_path = run / "ablation/thresholds" / (arm + ".json")
        th = e.json(th_path)
        require(th.get("status") == "validation_frozen" and th.get("manifest_sha256") == MANIFEST_SHA
                and th.get("evaluation_source_sha256") == CORE_SHA and th.get("config_sha256") == stable(cfg)
                and th.get("seeds") == list(SEEDS) and th.get("cases_per_seed") == 750
                and th.get("independent_parents") == 150 and set(th.get("thresholds", {})) == set(methods),
                "Arm threshold freeze differs: " + arm)
        require(analysis.get("status") == "complete_primary_reanalysis"
                and analysis.get("analysis_script_sha256") == script_sha
                and analysis.get("manifest_sha256") == MANIFEST_SHA and analysis.get("source_sha256") == CORE_SHA
                and analysis.get("config_sha256") == stable(cfg) and analysis.get("seeds") == list(SEEDS)
                and analysis.get("case_count") == 750 and analysis.get("methods") == list(methods),
                "Final raw reanalysis identity/coverage differs: " + arm)
        boot = analysis.get("bootstrap", {})
        require(boot.get("unit") == "generator_parent" and boot.get("replicates") == REPLICATES
                and boot.get("seed") == BOOTSTRAP_SEED, "Raw reanalysis bootstrap settings differ")
        provenance = analysis.get("evaluation_provenance", [])
        require(len(provenance) == 3 and {x["seed"] for x in provenance} == set(SEEDS), "Reanalysis provenance seeds")
        for n, seed in enumerate(SEEDS):
            f = run / "ablation/evaluations/test" / arm / ("seed" + str(seed))
            meta, intent = e.json(f / "evaluation.json"), e.json(f / "evaluation_intent.json")
            p = next(x for x in provenance if x["seed"] == seed)
            require(p == {"evaluation_id": "evaluation_" + str(n), "seed": seed,
                          "summary_sha256": e.digest(f / "evaluation.json"),
                          "intent_sha256": e.digest(f / "evaluation_intent.json"),
                          "per_case_sha256": e.digest(f / "per_case.csv")}, "Reanalysis provenance changed")
            require(meta.get("complete") is True and e.json(f / "failures.json") == []
                    and meta.get("journal_identity_sha256") == stable(intent), "Incomplete/mismatched evaluation")
            target = {"seed": seed, "split": "test", "limit": 0, "manifest_sha256": MANIFEST_SHA,
                      "source_sha256": CORE_SHA, "config_sha256": stable(cfg), "mask_seed": 42,
                      "num_holes": 8, "noise_rate": 0.0, "project_observed": False, "methods": list(methods),
                      "method_thresholds": th["thresholds"], "threshold_selection_sha256": e.digest(th_path),
                      "threshold_selection_status": "validation_frozen"}
            require(all(meta.get(k) == v and intent.get(k) == v for k, v in target.items()), "Evaluation settings differ")
            require(meta.get("expected_cases") == 750 and len(intent.get("case_ids", [])) == 750
                    and set(intent["case_ids"]) == set(parents) and intent.get("heldout_holes") == 4
                    and intent.get("samples_per_hole") == 12, "Evaluation cases/observations differ")
            for document in (meta, intent):
                pol = document.get("environment", {}).get("evaluation_numerical_policy", {})
                require(pol.get("declared") == NUMERICS and all(pol.get("actual", {}).get(k) == NUMERICS[k]
                        for k in ("cublas_workspace_config", "cudnn_benchmark", "cudnn_deterministic",
                                  "deterministic_algorithms", "deterministic_warn_only")), "Evaluation policy differs")
        rows = e.csv(folder / "recomputed_per_case.csv")
        expected_keys = {(seed, sid, method) for seed in SEEDS for sid in parents for method in methods}
        by_key = {}
        for row in rows:
            key = (int(row["seed"]), row["sample_id"], row["method"])
            require(key in expected_keys and key not in by_key, "Unexpected/duplicate recomputed row")
            require(row["parent_id"] == parents[key[1]] and row["mask_seed"] == "42"
                    and float(row["threshold_used"]) == float(th["thresholds"][key[2]]), "Recomputed identity differs")
            for metric in METRICS:
                value = float(row[metric])
                require(math.isfinite(value) and 0 <= value <= 1, "Invalid/missing metric: " + metric)
            by_key[key] = row
        require(set(by_key) == expected_keys, "Missing recomputed rows")
        # The attention_position adaptation rows are validated but not in cross-arm architecture contrasts.
        all_arms[arm] = {(s, sid): by_key[(s, sid, methods[0])] for s in SEEDS for sid in parents}
    e.raw(__file__)
    e.unchanged()
    return all_arms, parents, e


def compute(arms, parents):
    """Paired case differences, then equal-parent bootstrap conditional on three fits."""
    import numpy as np
    ids = sorted(set(parents.values()))
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(ids), size=(REPLICATES, len(ids)))
    contrasts, seed_rows, parent_rows = [], [], []
    for name, candidate, reference in CONTRASTS:
        for metric in METRICS:
            per_parent = {p: [] for p in ids}
            per_seed = {s: {p: [] for p in ids} for s in SEEDS}
            for seed in SEEDS:
                for sid, parent in sorted(parents.items()):
                    delta = float(arms[candidate][seed, sid][metric]) - float(arms[reference][seed, sid][metric])
                    per_parent[parent].append(delta); per_seed[seed][parent].append(delta)
            require(all(len(v) == 15 for v in per_parent.values()), "Invalid paired parent weighting")
            values = np.array([np.mean(per_parent[p]) for p in ids])
            boot = values[draws].mean(axis=1)
            contrasts.append({"contrast": name, "candidate": candidate, "reference": reference, "metric": metric,
                              "mean_candidate_minus_reference": float(values.mean()),
                              "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975)),
                              "independent_parents": len(ids), "paired_case_runs": len(parents) * 3})
            for seed in SEEDS:
                seed_rows.append({"contrast": name, "metric": metric, "seed": seed,
                                  "mean_candidate_minus_reference": float(np.mean([np.mean(per_seed[seed][p]) for p in ids]))})
            parent_rows.extend({"contrast": name, "metric": metric, "parent_id": p,
                                "mean_candidate_minus_reference": float(v)} for p, v in zip(ids, values))
    return contrasts, seed_rows, parent_rows, np.__version__


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run_analysis(run, out, check_only=False):
    out = Path(out).resolve()
    require(out != OUTPUT_BASE.resolve() and out.is_relative_to(OUTPUT_BASE.resolve()),
            "Output must be a new directory under admin/secondary_ablation_contrasts_outputs")
    require(not out.exists(), "Output already exists; refusing to overwrite")
    arms, parents, e = validate(run)
    if check_only:
        return {"status": "gates_passed_not_computed", "actual_ci_computed": False}
    estimates, seeds, groups, numpy_version = compute(arms, parents)
    e.unchanged()
    result = {"status": "complete_secondary_post_results_analysis", "not_preregistered": True,
              "interpretation": "Specified after viewing point estimates; all four contrasts and five metrics retained."
              " Pointwise 95% percentile intervals, not simultaneous or multiplicity-adjusted; no p-values,"
              " significance, equivalence or general architecture superiority inferred. Conditional on the three executed fits.",
              "metrics": list(METRICS), "error_threshold_scope": "observed_error and hidden_error use each arm's frozen threshold, not common 0.5",
              "bootstrap": {"replicates": REPLICATES, "seed": BOOTSTRAP_SEED, "unit": "generator_parent",
                            "parents": 150, "aggregation": "paired seed/sample difference, mean of five variants times three seeds per parent"},
              "software": {"python": platform.python_version(), "numpy": numpy_version}, "estimates": estimates,
              "input_sha256": {str(p): hashlib.sha256(data).hexdigest() for p, data in e.content.items()},
              "limitations": "Relies on final raw reanalysis and its provenance. Existing analysis.json does not checksum its recomputed CSV;"
              " this output binds those CSV bytes, but does not independently recompute predictions or validate array-level truth."}
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("contrasts.csv", estimates), ("seed_differences.csv", seeds), ("parent_differences.csv", groups)):
        write_csv(out / name, rows)
    result["output_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*.csv")}
    (out / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": result["status"], "output": str(out), "contrasts": len(estimates)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_analysis(args.run_root, args.out, args.check_only)
    except (Blocked, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "blocked_no_analysis", "reason": str(error), "actual_ci_computed": False}))
        return 2
    print(json.dumps(result)); return 0


if __name__ == "__main__":
    sys.exit(main())
