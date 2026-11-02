"""Fast protocol tests that never use the repository's raw data."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_data import (
    ProtocolDataset,
    audit_raw_sources,
    make_drill_masks,
    prepare_protocol,
    verify_protocol,
)


class ProtocolTests(unittest.TestCase):
    def test_parent_split_and_case_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = prepare_protocol(
                folder, str(Path(folder) / "protocol"), count=100,
                seed=731, grid_size=(16, 16, 16), audit_raw=False,
            )
            manifest = verify_protocol(manifest_path)
            self.assertEqual(manifest["split_counts"], {"train": 70, "val": 15, "test": 15})
            self.assertEqual(manifest["parent_counts"], {"train": 14, "val": 3, "test": 3})
            self.assertEqual(len({row["parent_id"] for row in manifest["samples"]}), 20)
            self.assertEqual(
                prepare_protocol(
                    folder, str(Path(folder) / "protocol"), count=100,
                    seed=731, grid_size=(16, 16, 16), audit_raw=False,
                ),
                manifest_path,
            )
            train = ProtocolDataset(manifest_path, "train")
            val = ProtocolDataset(manifest_path, "val")
            test = ProtocolDataset(manifest_path, "test")
            self.assertEqual((len(train), len(val), len(test)), (70, 15, 15))
            obs, target, case_id = train[0]
            self.assertEqual(tuple(obs.shape), (2, 16, 16, 16))
            self.assertEqual(tuple(target.shape), (16, 16, 16))
            self.assertEqual(set(np.unique(target.numpy())), {0.0, 1.0})
            self.assertGreater(obs[1].sum().item(), 0)
            self.assertTrue(np.array_equal(obs.numpy(), train[0][0].numpy()))
            case = train.get_case(case_id)
            self.assertFalse(np.any(case["observed_mask"] & case["hidden_mask"]))
            self.assertTrue(np.array_equal(
                case["obs"][0][case["observed_mask"]],
                case["target"][case["observed_mask"]],
            ))
            self.assertTrue(np.all(case["obs"][0][~case["observed_mask"]] == 0))
            self.assertEqual(case["metadata"]["domain"]["min"], [-1.0, -1.0, -1.0])
            flipped = []
            for rate in (0.05, 0.10, 0.20):
                noisy = ProtocolDataset(
                    manifest_path, "train", num_holes=8,
                    samples_per_hole=12, mask_seed=42, noise_rate=rate,
                ).get_case(case_id)
                self.assertTrue(np.array_equal(
                    noisy["observed_mask"], case["observed_mask"]))
                locations = np.flatnonzero(
                    noisy["obs"][0] != case["obs"][0])
                flipped.append(set(locations.tolist()))
            self.assertTrue(flipped[0] <= flipped[1] <= flipped[2])

    def test_hole_nesting_and_target_independence(self):
        small, hidden_small = make_drill_masks((16, 16, 16), "case-one", 2, 12, 42, 4)
        large, hidden_large = make_drill_masks((16, 16, 16), "case-one", 8, 12, 42, 4)
        self.assertTrue(np.all(small <= large))
        self.assertTrue(np.array_equal(hidden_small, hidden_large))
        self.assertFalse(np.any(large & hidden_large))
        self.assertTrue(np.array_equal(
            large,
            make_drill_masks((16, 16, 16), "case-one", 8, 12, 42, 4)[0],
        ))

    def test_raw_duplicate_audit_and_mesh_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "raw"
            (root / "mining_ply").mkdir(parents=True)
            (root / "mining_ply_generate").mkdir()
            (root / "mining_ply_generate_pretrain").mkdir()
            tetra = (
                "ply\nformat ascii 1.0\nelement vertex 4\n"
                "property float x\nproperty float y\nproperty float z\n"
                "element face 4\nproperty list uchar int vertex_indices\nend_header\n"
                "0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
                "3 0 1 2\n3 0 3 1\n3 1 3 2\n3 2 3 0\n"
            )
            (root / "mining_ply" / "a1.ply").write_text(tetra, encoding="utf-8")
            (root / "mining_ply_generate" / "generate_mine_e1.ply").write_text(tetra, encoding="utf-8")
            (root / "mining_ply_generate_pretrain" / "generate_mine_e1.ply").write_text(tetra, encoding="utf-8")
            audit = audit_raw_sources(root, Path(folder) / "audit")
            self.assertEqual(audit["legacy_5000_pattern_count"], 1)
            self.assertEqual(audit["pretrain_main_identical_sha256"], 1)
            self.assertEqual(audit["real_mesh_quality_status_counts"]["checked"], 1)
            records = [
                json.loads(line)
                for line in (Path(folder) / "audit" / "raw_inventory.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            real = next(row for row in records if row["source_role"] == "mining_ply")
            self.assertTrue(real["mesh_quality"]["watertight_by_edge_count"])


if __name__ == "__main__":
    unittest.main()

