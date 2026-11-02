"""SI showcase from saved formal seed11 predictions only; never runs inference.

The three reference-selected cases are immutable.  Missing/incomplete evidence
blocks every formal figure.  --layout-qa creates labelled toy layout fixtures
inside this helper's internal layout_qa directory, never scientific results.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np

HOME = Path(__file__).resolve().parent
SELECTION_SHA = "f9b6cee797e702ca9b7e26ab33af578c093fa5e26d5c7683b0b34ad0005861ab"
FIGURE_SHA = "3aaca3fb4bf51c653439b7c52557a405a5c29730479d70bf96d5ced4b629a3aa"
METHODS = ("idw", "rbf", "ik", "mps", "unet", "base_vae", "cgan", "geovae", "geovae_latent", "geovae_joint")
CASE_IDS = ("syn-p000736-v4", "syn-p000654-v1", "syn-p000651-v2")
WIDTH_MM = 130.75
LABELS = ("Curved main lens", "Auxiliary-lobe union", "Disconnected voxel region")


class EvidenceUnavailable(RuntimeError):
    """A required completed artifact is absent; no scientific substitute exists."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read(path):
    path = Path(path)
    if not path.is_file():
        raise EvidenceUnavailable("Required evidence is absent: " + path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def output_path(path, draft=False):
    out = Path(path).resolve()
    if not out.is_relative_to(HOME) or out == HOME:
        raise ValueError("This independent helper writes only a subdirectory of its internal public_cases folder")
    if draft and not out.is_relative_to(HOME / "layout_qa"):
        raise ValueError("Toy layout fixtures must remain inside internal layout_qa")
    return out


def sources(project):
    path = Path(project) / "revision_figures.py"
    if not path.is_file() or sha(path) != FIGURE_SHA:
        raise ValueError("The reused frozen plotting helper does not match its approved hash")
    spec = importlib.util.spec_from_file_location("frozen_showcase_style", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                                "pdf.fonttype": 42, "svg.fonttype": "none"})
    return module


def source_functions(path):
    names = {"_stable_seed", "make_drill_masks", "generate_occupancy"}
    nodes = [node for node in ast.parse(Path(path).read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("Required archived geometric/sampling functions are absent")
    namespace = {"np": np, "math": math, "hashlib": hashlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def header(identity, meta, intent, thresholds, manifest_sha, expected_ids):
    if identity.get("mode") != "formal" or identity.get("limit") != 0:
        raise EvidenceUnavailable("A formal unlimited run is required")
    if not meta.get("complete") or meta.get("evaluation_status") != "formal_full":
        raise EvidenceUnavailable("The complete formal primary seed11 evaluation is not available")
    expected = {"split": "test", "seed": 11, "mask_seed": 42, "num_holes": 8,
                "noise_rate": 0., "limit": 0, "project_observed": False,
                "threshold_selection_status": "validation_frozen", "manifest_sha256": manifest_sha}
    for key, value in expected.items():
        if meta.get(key) != value or intent.get(key) != value:
            raise ValueError("Evaluation condition is incompatible: " + key)
    if intent.get("samples_per_hole") != 12 or intent.get("heldout_holes") != 4:
        raise ValueError("The saved evaluation does not use the registered main observation budget")
    if meta.get("expected_cases") != 750 or set(intent.get("case_ids", [])) != set(expected_ids) or len(intent.get("case_ids", [])) != 750:
        raise EvidenceUnavailable("Exactly all750 registered test cases are required")
    if meta.get("case_ids_sha256") != stable(intent["case_ids"]):
        raise ValueError("Evaluation case-list identity differs")
    if set(meta.get("methods", [])) != set(METHODS) or set(intent.get("methods", [])) != set(METHODS):
        raise EvidenceUnavailable("All ten registered methods are required")
    if intent.get("case_selection_sha256") is not None or meta.get("case_selection_sha256") is not None:
        raise ValueError("Subset evaluations cannot supply this SI showcase")
    if identity.get("manifest_sha256") != manifest_sha or thresholds.get("manifest_sha256") != manifest_sha:
        raise ValueError("Run/threshold and reference manifest identities differ")
    if thresholds.get("status") != "validation_frozen":
        raise EvidenceUnavailable("Formal validation-frozen thresholds are required")
    for key in ("config_sha256", "source_sha256", "checkpoints", "method_thresholds"):
        if meta.get(key) != intent.get(key):
            raise ValueError("Evaluation metadata/intent differs: " + key)
    if (thresholds.get("config_sha256") != intent["config_sha256"]
            or thresholds.get("evaluation_source_sha256") != intent["source_sha256"]
            or identity.get("source_sha256") != intent["source_sha256"]):
        raise ValueError("Threshold, run and evaluation source/config identities differ")
    if any(float(thresholds["thresholds"][method]) != float(intent["method_thresholds"][method])
           or not 0 < float(intent["method_thresholds"][method]) < 1 for method in METHODS):
        raise ValueError("Per-method thresholds do not match their frozen selection")
    numerical = meta.get("environment", {}).get("evaluation_numerical_policy", {})
    actual, declared = numerical.get("actual", {}), numerical.get("declared", {})
    for key, value in {"cublas_workspace_config": ":4096:8", "deterministic_algorithms": True,
                       "deterministic_warn_only": False, "cudnn_benchmark": False,
                       "cudnn_deterministic": True}.items():
        if actual.get(key) != value or declared.get(key) != value:
            raise ValueError("The v2 strict numerical-policy evidence differs: " + key)
    if meta.get("environment") != intent.get("environment"):
        raise ValueError("Evaluation environment identity differs")
    expected_sha = stable(intent)
    if meta.get("journal_identity_sha256") != expected_sha:
        raise ValueError("Evaluation intent/journal identity differs")
    return expected_sha


def prediction(folder, record, sample, parent, method, threshold, identity_sha):
    journal_path = folder / "journal" / method / (sample + ".json")
    journal = read(journal_path)
    if (journal.get("status") != "complete" or journal.get("evaluation_identity_sha256") != identity_sha
            or journal.get("sample_id") != sample or journal.get("method") != method):
        raise EvidenceUnavailable("A selected-case journal is absent, incomplete or incompatible")
    stored = journal["record"]
    for key, value in {"sample_id": sample, "parent_id": parent, "method": method, "split": "test"}.items():
        if stored.get(key) != value or record.get(key) != value:
            raise ValueError("Selected CSV/journal case identity differs: " + key)
    for key, value in {"seed": 11, "mask_seed": 42, "num_holes": 8, "noise_rate": 0.}.items():
        if float(stored.get(key, -1)) != value or float(record.get(key, -1)) != value:
            raise ValueError("Selected prediction condition differs: " + key)
    if float(record["threshold_used"]) != threshold or float(stored["threshold_used"]) != threshold:
        raise ValueError("Selected prediction threshold differs")
    expected = "predictions/" + method + "/" + sample + ".npy"
    if (stored.get("prediction_file") != expected or record.get("prediction_file") != expected
            or record.get("prediction_sha256") != stored.get("prediction_sha256")):
        raise ValueError("Selected prediction path/hash identity differs")
    path = (folder / expected).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise EvidenceUnavailable("Selected saved prediction is absent or escapes its evaluation")
    if sha(path) != stored["prediction_sha256"]:
        raise ValueError("Selected saved prediction checksum differs")
    probability = np.load(path, allow_pickle=False)
    if probability.shape != (32, 32, 32) or not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError("Selected saved probabilities have invalid shape/range")
    return probability, path, journal_path


def load_evidence(run, selection_path, style):
    run = Path(run).resolve()
    if (run / "smoke_identity.json").exists():
        raise EvidenceUnavailable("CPU smoke evidence cannot become formal SI figures")
    if sha(selection_path) != SELECTION_SHA:
        raise ValueError("The immutable three-case selection has changed")
    selection = read(selection_path)
    if tuple(s["native_reference_evidence"]["sample_id"] for s in selection["selection"]) != CASE_IDS:
        raise ValueError("Frozen case IDs or order differs")
    frozen, folder = run / "frozen_inputs", run / "test" / "seed11"
    manifest_path, threshold_path = frozen / "manifest.json", run / "thresholds.json"
    manifest = read(manifest_path)
    if (manifest.get("protocol_version") != "procedural-lens-v1" or manifest.get("source_data_policy") != "independent_procedural_only_no_real_mesh_parents"
            or manifest.get("grid_size") != [32, 32, 32] or sha(manifest_path) != selection["source_sha256"]["manifest.json"]):
        raise ValueError("Only the frozen independent synthetic protocol is allowed")
    entries = {row["sample_id"]: row for row in manifest["samples"] if row["split"] == "test"}
    identity, meta, intent, thresholds = (read(run / "run_identity.json"), read(folder / "evaluation.json"),
                                        read(folder / "evaluation_intent.json"), read(threshold_path))
    identity_sha = header(identity, meta, intent, thresholds, sha(manifest_path), entries)
    training_path = run / "models" / "geovae" / "seed11" / "run.json"
    training = read(training_path)
    if (training.get("manifest_sha256") != sha(manifest_path)
            or training.get("config", {}).get("latent_dim") != 256
            or training.get("config", {}).get("grid_size") != [32, 32, 32]):
        raise ValueError("The retained main GeoVAE must be the registered d256/32-cubed fit")
    if sha(threshold_path) != intent.get("threshold_selection_sha256") or meta.get("threshold_selection_sha256") != sha(threshold_path):
        raise ValueError("Validation threshold file checksum differs")
    if read(folder / "failures.json"):
        raise EvidenceUnavailable("The primary formal evaluation still has failed cases")
    table_path = folder / "per_case.csv"
    if not table_path.is_file():
        raise EvidenceUnavailable("The complete primary per-case table is absent")
    selected_rows = {}
    keys = set()
    with table_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["sample_id"], row["method"])
            if key in keys or key[0] not in entries or key[1] not in METHODS:
                raise ValueError("Duplicate/unregistered row in the complete primary table")
            keys.add(key)
            if key[0] in CASE_IDS:
                selected_rows[key] = row
    if len(keys) != 750 * len(METHODS):
        raise EvidenceUnavailable("The complete750case×10method primary table is required")
    generator = frozen / "experiment_data.py"
    if sha(generator) != selection["source_sha256"]["experiment_data.py"]:
        raise ValueError("Archived generator/sampling implementation differs")
    functions = source_functions(generator)
    sources = [selection_path, manifest_path, generator, run / "run_identity.json", folder / "evaluation.json",
               folder / "evaluation_intent.json", threshold_path, folder / "failures.json", table_path, training_path]
    cases = []
    for index, selected in enumerate(selection["selection"]):
        sample = CASE_IDS[index]; row = entries[sample]
        if row.get("source") != "independent_procedural_lens" or row.get("source_parent") is not None:
            raise ValueError("Real/legacy generated geometry cannot enter this figure")
        grid = (frozen / row["path"]).resolve()
        evidence = selected["native_reference_evidence"]
        if (not grid.is_relative_to(frozen.resolve()) or sha(grid) != row["sha256"]
                or row["sha256"] != evidence["target_file_sha256"]):
            raise ValueError("Selected target file identity differs")
        target = np.load(grid, allow_pickle=False)
        if (target.shape != (32, 32, 32) or target.dtype != np.uint8
                or hashlib.sha256(target.tobytes()).hexdigest() != evidence["target_occupancy_sha256"]):
            raise ValueError("Selected target occupancy identity differs")
        obs, hidden = functions["make_drill_masks"](target.shape, sample, 8, 12, 42, 4)
        observation = np.stack((target * obs, obs)).astype(np.float32)
        if int(target[obs].sum()) != (0, 1, 1)[index] or int(obs.sum()) != 96:
            raise ValueError("Frozen0/1/1positive-observation evidence differs")
        plane = style._plane(observation)
        if style._plane_record(plane) != evidence["fixed_plane"]:
            raise ValueError("The reference-selected common plane has changed")
        without_params = dict(row["params"]); without_params["branch"] = False
        without = functions["generate_occupancy"](target.shape, without_params)
        added = (target > 0) & (without == 0)
        if int(added.sum()) != evidence["auxiliary_lobe_added_voxels"] or np.any(added & obs):
            raise ValueError("Frozen auxiliary-lobe evidence differs")
        predictions, thresholds_used = {}, {}
        for method in METHODS:
            threshold = float(intent["method_thresholds"][method])
            probability, path, journal = prediction(folder, selected_rows[(sample, method)], sample,
                                                    row["parent_id"], method, threshold, identity_sha)
            predictions[method] = probability >= threshold
            thresholds_used[method] = threshold
            sources.extend((path, journal))
        cases.append({"sample_id": sample, "parent_id": row["parent_id"], "role": selected["role"],
                      "label": LABELS[index], "target": target, "added": added, "observation": observation,
                      "plane": plane, "predictions": predictions, "thresholds": thresholds_used,
                      "reference_evidence": evidence})
        sources.append(grid)
    return cases, sources, {"evaluation_identity_sha256": identity_sha, "model_seed": 11,
                            "threshold_source": "validation_frozen; each method uses its saved threshold",
                            "project_observed": False, "selected_case_count": 3,
                            "primary_test_count": 750, "primary_test_parents": 150,
                            "evaluation_checkpoint_identities": intent["checkpoints"],
                            "numerical_policy": meta["environment"]["evaluation_numerical_policy"]}


def qa_tools():
    names = ("GEOVAE_FIGURE_ALIGNMENT_AUDITOR", "GEOVAE_FIGURE_COLLISION_AUDITOR", "GEOVAE_FIGURE_FONT_AUDITOR")
    if any(not os.environ.get(name) or not Path(os.environ[name]).is_file() for name in names):
        raise EvidenceUnavailable("The three existing alignment/collision/font QA tool paths must be supplied before export")


def save(fig, out, name, style, contract, source_files, draft):
    qa_tools()
    # Keep this independent helper's alignment gate explicit as well as the
    # inherited exporter's gate; both measure the final physical plot areas.
    path = os.environ["GEOVAE_FIGURE_ALIGNMENT_AUDITOR"]
    spec = importlib.util.spec_from_file_location("showcase_alignment", path)
    auditor = importlib.util.module_from_spec(spec); sys.modules[spec.name] = auditor
    spec.loader.exec_module(auditor)
    options = style._alignment([ax for ax in fig.axes if ax.get_visible()], 4 if len(fig.axes) == 12 else 3)
    # Editable PDF/SVG remain authoritative; TIFF is the600dpi raster companion.
    if draft:
        fig.text(.5, .985, "LAYOUT FIXTURE | invented geometry/panel fills; not experimental results",
                 ha="center", va="top", fontsize=6.5, color="#883344")
    fig.canvas.draw()
    auditor.require_matplotlib_panel_alignment(fig, json_out=out/(name+".independent_alignment.json"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, **options)
    style.export(fig, out, name, contract, source_files,
                 alignment_options=options,
                 draft=False)
    # export closes the pyplot manager, but this Figure remains renderable.
    fig.savefig(out / (name + ".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    provenance_path = out / (name + ".provenance.json")
    provenance = read(provenance_path)
    provenance.update({"evidence_role": "layout_fixture_not_experimental" if draft else "illustrative_subset_of_completed_primary_test",
                       "selection_sha256": SELECTION_SHA, "independent_plot_helper_sha256": sha(__file__),
                       "reused_plot_helper_sha256": FIGURE_SHA,
                       "reuse_level": "Structural adaptation of frozen common-plane/annotation/export primitives",
                       "exports": {suffix: sha(out / (name + suffix)) for suffix in (".pdf", ".svg", ".tiff")},
                       "selection_limit": "Three reference-selected examples from the existing750/150parenttest; not an external test",
                       "no_inference": True})
    write(provenance_path, provenance)


def render(cases, out, style, source_files, metadata, draft=False):
    plt = style.plt
    out.mkdir(parents=True, exist_ok=True)
    contracts = []
    for index, case in enumerate(cases):
        plane = case["plane"]
        truth = style._section(case["target"], plane)
        extension = style._section(case["added"].astype(np.uint8), plane)
        panels = [("Reference\noccupancy", truth), ("Observed\nlabels only", None)]
        for method in METHODS:
            name = style.DISPLAY[method].replace("GeoVAE + ", "GeoVAE\n+ ")
            name = name.replace("Indicator kriging", "Indicator\nkriging").replace("MPS (GENESIM)", "MPS\n(GENESIM)")
            panels.append((name, style._section(case["predictions"][method].astype(np.uint8), plane)))
        fig, grid = plt.subplots(3, 4, figsize=(WIDTH_MM / 25.4, 100 / 25.4))
        fig.subplots_adjust(left=.065, right=.98, top=.81, bottom=.24, hspace=.69, wspace=.48)
        axes = list(grid.ravel())
        for ax, (title, image) in zip(axes, panels):
            style._plane_panel(ax, plane, truth, title, image)
            if image is not None and np.any(extension) and np.any(extension == 0):
                ax.contour(plane["u"], np.arange(truth.shape[0]), extension, levels=[.5],
                           colors="#CC8833", linewidths=.7)
        for ax in axes[:8]:
            ax.tick_params(labelbottom=False)
        style._panel_labels(axes)
        fig.text(.5, .925, case["label"] + " | " + case["sample_id"], ha="center", fontsize=7)
        positives = int(np.count_nonzero(case["observation"][0]))
        conditions=(f"Same mask/plane; {positives}/96 observed occupied labels; seed11; method-specific validation thresholds.\n"
                    "Dashed: reference. Orange: reference auxiliary-lobe ROI, not a predicted feature label." if not draft else
                    "Toy panel fills only; no model fits, validation thresholds or inference.\n"
                    "Dashed: toy reference. Orange: toy reference extension, not a model feature label.")
        fig.text(.5, .153, conditions,
                 ha="center", fontsize=6.1, linespacing=1.2)
        fig.text(.5, .067, "Auxiliary-lobe voxels are unobserved. These examples do not establish external generalization.\n"
                 "Native32-cubed index grid; individual descriptive images, no uncertainty interval.",
                 ha="center", fontsize=6.1, linespacing=1.2)
        contract = {"question": "What are the saved primary predictions for a prespecified complex reference example?",
                    "claim": "Identical inputs, domain and reference ROI are compared without assuming feature recovery.",
                    "archetype": "image plate + quant", "data_type": "actual_saved_predictions" if not draft else "layout_fixture",
                    "sample_id": case["sample_id"], "parent_id": case["parent_id"], "role": case["role"],
                    "domain": "Identical32³index grid; voxel centers0..31; physical coordinate units not asserted",
                    "plane": style._plane_record(plane), "observed_occupied_labels": positives,
                    "observed_total": 96, "observed_auxiliary_lobe_labels": 0,
                    "thresholds": case["thresholds"], "seed": 11, "methods": list(METHODS),
                    "exclusions": "None; missing evidence blocks the complete figure bundle", **metadata}
        save(fig, out, f"complex_case_{index+1}_sections", style, contract, source_files, draft)
        contracts.append(contract)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 65 / 25.4))
    axes = [fig.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    fig.subplots_adjust(left=.015, right=.985, top=.85, bottom=.27, wspace=.04)
    coordinates = np.indices((33, 33, 33), dtype=float) - .5
    for ax, case, index in zip(axes, cases, range(3)):
        occupied = case["target"].transpose(2, 1, 0).astype(bool)
        extension = case["added"].transpose(2, 1, 0).astype(bool)
        colors = np.empty(occupied.shape, dtype=object); colors[:] = "#4477AA"; colors[extension] = "#CC8833"
        ax.voxels(*coordinates, occupied, facecolors=colors, edgecolor="#555555", linewidth=.10)
        ax.set(xlim=(-.5,31.5), ylim=(-.5,31.5), zlim=(-.5,31.5))
        ax.set_box_aspect((1,1,1)); ax.view_init(elev=22, azim=-60); ax.set_axis_off()
        ax.set_title(chr(97+index) + "  " + case["label"].replace(" ", "\n", 1), fontsize=7, pad=3)
    fig.text(.5,.145,"References only; identical index domain and camera. Blue: main lens; orange: auxiliary-lobe extension.",
             ha="center",fontsize=6.2)
    observed_text=", ".join(str(int(np.count_nonzero(case["observation"][0])))+"/96" for case in cases)
    fig.text(.5,.055,("Reference-selected members of the existing test cohort; native32-cubed geometry.\n" if not draft
                    else "Toy reference fixtures; native32-cubed geometry; not selected scientific cases.\n")
             +"Observed occupied labels:"+observed_text+". No auxiliary-lobe voxel is directly observed.",
             ha="center",fontsize=6.2,linespacing=1.2)
    save(fig,out,"complex_reference_geometries",style,{"question":"Which native reference features motivated these illustrative examples?",
        "claim":"The recorded generator/voxel features are shown without geological fault or feature-recovery interpretation.",
        "archetype":"image plate + quant","data_type":"actual_reference_geometry" if not draft else "layout_fixture",
        "camera":{"elevation":22,"azimuth":-60},"colors":"Blue main lens; orange newly occupied auxiliary-lobe voxels",
        "sample_ids":[case["sample_id"] for case in cases],"case_count":3,**metadata},source_files,draft)
    write(out/"figure_contracts.json",contracts)


def layout_cases(style):
    """Explicit toy fixtures for layout only; no paper IDs or performance values."""
    zz,yy,xx=np.indices((32,32,32))
    target=(((xx-16)/8)**2+((yy-16)/7)**2+((zz-15)/3)**2<1)
    extra=(((xx-14)/4)**2+((yy-16)/4)**2+((zz-8)/2)**2<1)&~target
    targets=[target,target|extra,target|np.roll(extra,-4,axis=0)]
    cases=[]
    for index,truth in enumerate(targets):
        observation=np.zeros((2,32,32,32),dtype=np.float32)
        for y,x in ((7,6),(27,25),(2,3),(29,3),(4,27),(25,6),(5,14),(26,28)):
            observation[1,np.rint(np.linspace(0,31,12)).astype(int),y,x]=1
        observation[0]=truth*observation[1]
        plane=style._plane(observation)
        predictions={method:(truth.copy() if j%3==0 else np.roll(truth,1,axis=1) if j%3==1 else np.zeros_like(truth))
                     for j,method in enumerate(METHODS)}
        cases.append({"sample_id":f"TOY-LAYOUT-{index+1}","parent_id":f"TOY-{index+1}",
            "label":"Layout fixture "+str(index+1),"role":"layout_fixture","target":truth.astype(np.uint8),
            "added":truth&~target,"observation":observation,"plane":plane,"predictions":predictions,
            "thresholds":{method:.5 for method in METHODS}})
    return cases


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=HOME.parents[2]/"gpt_thinking/Project_VAE/3dvae")
    parser.add_argument("--run",type=Path)
    parser.add_argument("--selection",type=Path,default=HOME/"complex_synthetic_selection.json")
    parser.add_argument("--out",type=Path,default=HOME/"formal_figures")
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--layout-qa",action="store_true")
    args=parser.parse_args(argv)
    out=output_path(args.out,args.layout_qa)
    try:
        style=sources(args.project)
        if args.layout_qa:
            cases=layout_cases(style);files=[];metadata={"evidence":"invented toy fixtures only; not experimental results"}
        else:
            if args.run is None: raise EvidenceUnavailable("Provide the completed formal run using --run")
            cases,files,metadata=load_evidence(args.run,args.selection,style)
        if args.check_only:
            out.mkdir(parents=True,exist_ok=True)
            write(out/"readiness.json",{"status":"ready","checked_at_utc":datetime.now(timezone.utc).isoformat(),"case_count":len(cases),**metadata})
            return 0
        qa_tools()
        out.mkdir(parents=True,exist_ok=True)
        write(out/"readiness.json",{"status":"rendering_not_complete","helper_sha256":sha(__file__),
              "layout_fixture":args.layout_qa,"checked_at_utc":datetime.now(timezone.utc).isoformat()})
        render(cases,out,style,files,metadata,args.layout_qa)
        write(out/"readiness.json",{"status":"layout_fixture_complete" if args.layout_qa else "formal_figures_complete",
             "checked_at_utc":datetime.now(timezone.utc).isoformat(),"helper_sha256":sha(__file__),**metadata})
        return 0
    except Exception as error:
        out.mkdir(parents=True,exist_ok=True)
        write(out/"readiness.json",{"status":"blocked_no_formal_figure_export","error_type":type(error).__name__,
             "reason":str(error),"checked_at_utc":datetime.now(timezone.utc).isoformat(),"helper_sha256":sha(__file__),
             "exports_complete":False,"partial_or_stale_assets_require_discard":any(out.glob('*.pdf'))})
        print("[SHOWCASE BLOCKED] "+str(error),file=sys.stderr)
        return 2


if __name__=="__main__":raise SystemExit(main())
