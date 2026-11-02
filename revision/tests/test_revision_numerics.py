"""CPU policy checks. These tests never initialize CUDA or run GPU training."""
from __future__ import annotations

import json
import sys
import types

import pytest
import torch
import torch.nn.functional as F

import repro_pipeline
import revision_numerics as numerics
from revision_integrity import AuditError, audit_numerical_policy


@pytest.fixture(autouse=True)
def restore_backend_flags():
    old = numerics.actual_flags()
    threads = torch.get_num_threads()
    yield
    torch.use_deterministic_algorithms(old["deterministic_algorithms"], warn_only=old["deterministic_warn_only"])
    torch.backends.cudnn.benchmark = old["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = old["cudnn_deterministic"]
    torch.set_num_threads(threads)


def test_configures_strict_evaluation_before_cuda_and_writes_phase_proof(tmp_path, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    proof = numerics.configure_evaluation(tmp_path, "validation/seed11")
    assert proof["kind"] == "evaluation" and proof["actual"]["deterministic_algorithms"]
    assert not proof["actual"]["deterministic_warn_only"]
    assert proof["actual"]["cublas_workspace_config"] == ":4096:8"
    assert not proof["actual"]["cudnn_benchmark"] and proof["actual"]["cudnn_deterministic"]
    assert not proof["cuda_initialized_at_configuration"]
    assert proof["workspace_origin"] == "configured_before_cuda_initialization"
    files = list((tmp_path / "numerical_policy").glob("*.json"))
    assert len(files) == 1 and files[0].name.startswith("validation_seed11_pid")
    assert json.loads(files[0].read_text()) == proof
    assert str(tmp_path) not in json.dumps(proof)
    assert not proof["training_bitwise_reproducibility_claimed"]


def test_conflicting_explicit_cublas_policy_is_rejected_without_changing_flags(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    before = numerics.actual_flags()
    with pytest.raises(RuntimeError, match="Incompatible"):
        numerics.configure_evaluation()
    assert numerics.actual_flags() == before


def test_missing_workspace_after_cuda_initialization_is_rejected(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA"):
        numerics.configure_evaluation()


def test_compatible_explicit_workspace_is_recorded(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    proof = numerics.configure_evaluation()
    assert proof["workspace_origin"] == "preexisting_compatible"
    assert proof["cuda_initialized_at_configuration"]


def test_metadata_adapter_is_idempotent_preserves_base_fields_and_imported_aliases(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    base = getattr(repro_pipeline.environment_info, "_revision_numerical_base", repro_pipeline.environment_info)
    monkeypatch.setattr(repro_pipeline, "environment_info", base)
    fake_module = types.ModuleType("_revision_numerics_alias_fixture")
    fake_module.environment_info = base
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    old = base()
    numerics.configure_evaluation()
    wrapped = repro_pipeline.environment_info
    assert fake_module.environment_info is wrapped
    new = wrapped()
    assert {key: value for key, value in new.items() if key != "evaluation_numerical_policy"} == old
    assert new["evaluation_numerical_policy"] == numerics.metadata()
    numerics.configure_evaluation()
    assert repro_pipeline.environment_info is wrapped


def test_thread_count_does_not_change_prediction_cache_environment(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    numerics.configure_evaluation()
    torch.set_num_threads(1)
    single = repro_pipeline.environment_info()
    torch.set_num_threads(4)
    multiple = repro_pipeline.environment_info()
    assert single == multiple
    assert "thread" not in json.dumps(single["evaluation_numerical_policy"]).lower()


def test_configuration_does_not_replace_losses_models_loader_or_rng(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    prediction, loader, loss = repro_pipeline._predict, repro_pipeline._load_checked_model, F.binary_cross_entropy_with_logits
    source = repro_pipeline.source_hash()
    rng = torch.random.get_rng_state().clone()
    logits, target = torch.tensor([-2., .5, 3.]), torch.tensor([0., 1., 1.])
    expected = loss(logits, target)
    numerics.configure_evaluation()
    assert repro_pipeline._predict is prediction and repro_pipeline._load_checked_model is loader
    assert F.binary_cross_entropy_with_logits is loss
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(loss(logits, target), expected)
    assert repro_pipeline.source_hash() == source


def test_training_record_is_separate_and_does_not_change_recipe(tmp_path, monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch.use_deterministic_algorithms(False)
    before = numerics.actual_flags()
    proof = numerics.record_training_policy(tmp_path, "train/geovae/seed11")
    assert proof["kind"] == "training" and not proof["actual"]["deterministic_algorithms"]
    assert numerics.actual_flags() == before
    assert not proof["training_bitwise_reproducibility_claimed"]
    numerics.configure_evaluation()
    with pytest.raises(RuntimeError, match="fresh unchanged training"):
        numerics.record_training_policy()


def test_final_policy_audit_requires_training_record_and_phase_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(False)
    numerics.record_training_policy(tmp_path, "train")
    numerics.configure_evaluation(tmp_path, "validation")
    numerics.configure_evaluation(tmp_path, "test")
    identity = {"evaluation_numerical_policy": numerics.DECLARED_POLICY,
                "evaluation_numerics_source_sha256": numerics.helper_sha256()}
    result = audit_numerical_policy(tmp_path, identity, required_phases=("validation", "test"))
    assert result["evaluation_phase_count"] == 2 and result["training_phase_count"] == 1
    path = next((tmp_path / "numerical_policy").glob("test_pid*.json"))
    proof = json.loads(path.read_text())
    proof["actual"]["deterministic_algorithms"] = False
    path.write_text(json.dumps(proof))
    with pytest.raises(AuditError, match="numerical_flags"):
        audit_numerical_policy(tmp_path, identity, required_phases=("validation", "test"))
