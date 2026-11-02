"""Known-answer, independence and provenance checks for revision statistics."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from experiment_data import prepare_protocol
from repro_metrics import surface_distances, volume_metrics
from revision_analysis import analyse_evaluations, nested_drill_conditions, parent_bootstrap


def test_surface_distance_and_dice_known_answer():
    truth = np.zeros((5, 5, 5), dtype=np.float32)
    predicted = truth.copy()
    truth[1, 1, 1] = 1
    predicted[3, 1, 1] = 1
    metrics = volume_metrics(predicted, truth, np.zeros_like(truth))
    assert metrics["hd"] == metrics["hd95"] == metrics["chamfer"] == 2
    assert metrics["overall_dice"] == metrics["blind_iou"] == 0
    assert metrics["surface_distance_status"] == "valid"
    scaled = surface_distances(predicted, truth, spacing=(.5, 1, 1))
    assert scaled["hd"] == scaled["chamfer"] == 1


def test_empty_surfaces_are_undefined_and_counted_not_zero():
    empty = np.zeros((4, 4, 4), dtype=np.float32)
    one = empty.copy()
    one[1, 1, 1] = 1
    missed = volume_metrics(empty, one, empty)
    assert missed["hd"] is None and missed["hd95"] is None and missed["chamfer"] is None
    assert missed["empty_prediction"] and not missed["surface_distance_valid"]
    assert missed["surface_distance_status"] == "prediction_empty"
    both = volume_metrics(empty, empty, empty)
    assert both["blind_iou"] == both["overall_dice"] == 1
    assert both["surface_distance_status"] == "both_empty"
    assert both["hd"] is None


def test_observed_hidden_and_volume_errors_known_answer():
    truth = np.zeros((4, 4, 4), dtype=np.float32)
    truth[1, 1, 1] = 1
    p = truth.copy()
    p[1, 1, 1] = .25
    known = np.zeros_like(truth, dtype=bool)
    known[1, 1, 1] = True
    hidden = known.copy()  # Overlap must not masquerade as hidden evidence.
    metrics = volume_metrics(p, truth, known, hidden_mask=hidden)
    assert metrics["observed_violation_count"] == 1
    assert metrics["observed_mse_truth"] == .75 ** 2
    assert metrics["volume_relative_error"] == -1
    assert metrics["volume_absolute_relative_error"] == 1
    assert metrics["hidden_n"] == 0 and metrics["hidden_error"] is None
    with pytest.raises(ValueError, match="probabilities"):
        volume_metrics(np.full_like(truth, 1.1), truth, known)
    with pytest.raises(ValueError, match="spacing"):
        surface_distances(truth, truth, spacing=(1, 0, 1))


def test_parent_bootstrap_does_not_weight_repeated_variants_as_independent():
    rows = [{"parent_id": "a", "score": 0.0}] * 100 + [{"parent_id": "b", "score": 1.0}]
    result = parent_bootstrap(rows, "score", replicates=2000, seed=1)
    dedup = parent_bootstrap([rows[0], rows[-1]], "score", replicates=2000, seed=1)
    assert result["mean"] == .5
    assert result["independent_parents"] == 2
    assert result["valid_case_runs"] == 101
    assert result["ci_low"] == dedup["ci_low"] and result["ci_high"] == dedup["ci_high"]
    assert parent_bootstrap([], "score")["mean"] is None


def test_retention_masks_are_nested_and_hidden_holes_unchanged():
    masks = nested_drill_conditions((32, 32, 32), "fixed_case")
    previous = np.zeros((32, 32, 32), dtype=bool)
    hidden_reference = masks[2][1]
    for holes in (2, 4, 6, 8, 10):
        observed, hidden = masks[holes]
        assert observed.sum() == holes * 12
        assert hidden.sum() == 4 * 12
        assert not np.any(previous & ~observed)
        assert not np.any(observed & hidden)
        np.testing.assert_array_equal(hidden, hidden_reference)
        previous = observed
    again = nested_drill_conditions((32, 32, 32), "fixed_case")
    np.testing.assert_array_equal(masks[8][0], again[8][0])


@pytest.fixture
def tiny_evaluation(tmp_path):
    from repro_pipeline import evaluate
    manifest = prepare_protocol(str(tmp_path), str(tmp_path / "protocol"), count=20,
                                seed=42, grid_size=(16, 16, 16), audit_raw=False)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"grid_size": [16] * 3, "mask_seed": 42,
                                  "num_holes": 2, "samples_per_hole": 4,
                                  "device": "cpu", "threshold": .5,
                                  "latent_dim": 16}), encoding="utf-8")
    directory = tmp_path / "evaluation"
    evaluate(manifest, str(config), str(directory), "test", ["idw", "rbf"], {},
             seed=11, limit=2)
    return manifest, directory, tmp_path


def test_saved_prediction_reanalysis_and_classical_duplicate_count(tiny_evaluation):
    manifest, directory, root = tiny_evaluation
    result = analyse_evaluations(manifest, [directory], root / "analysis", formal=False,
                                required_methods=("idw", "rbf"), replicates=20)
    assert result.is_file()
    with result.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["independent_model_seeds"] == "0"
    assert rows[0]["unique_cases"] == "2"
    meta = json.loads((result.parent / "analysis.json").read_text())
    assert meta["status"] == "draft_reanalysis"
    assert meta["distance_unit"] == "voxel"
    assert str(root) not in (result.parent / "analysis.json").read_text()


def test_reanalysis_rejects_tampered_csv_and_prediction(tiny_evaluation):
    manifest, directory, root = tiny_evaluation
    path = directory / "per_case.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    old = rows[0]["blind_iou"]
    rows[0]["blind_iou"] = str(float(old) + .1)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(RuntimeError, match="recomputed"):
        analyse_evaluations(manifest, [directory], root / "analysis", formal=False,
                            required_methods=("idw", "rbf"), replicates=20)
    assert json.loads((root / "analysis" / "analysis.json").read_text())["status"] == "running"
    rows[0]["blind_iou"] = old
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    prediction = directory / rows[0]["prediction_file"]
    p = np.load(prediction)
    p.flat[0] = 1 - p.flat[0]
    np.save(prediction, p)
    with pytest.raises(RuntimeError, match="checksum"):
        analyse_evaluations(manifest, [directory], root / "analysis", formal=False,
                            required_methods=("idw", "rbf"), replicates=20)


def test_tiny_diagnostic_can_never_become_formal(tiny_evaluation):
    manifest, directory, root = tiny_evaluation
    with pytest.raises(ValueError, match="5000-case"):
        analyse_evaluations(manifest, [directory], root / "analysis", formal=True,
                            required_methods=("idw", "rbf"))
