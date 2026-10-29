"""Small path/argument fixtures only; no models, source data or grids are used."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repository_safety import validate_arguments


def option(arguments, flag):
    return arguments[arguments.index(flag) + 1]


class RepositorySafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.repo = self.base / "repository"
        (self.repo / "revision").mkdir(parents=True)
        self.outside = self.base / "private"
        self.source = self.base / "source"

    def real_args(self, output):
        return ["--real-cases", "--data-root", str(self.source), "--private-out", str(output)]

    def external_args(self, command="prepare"):
        if command == "prepare":
            return [command, "--source", str(self.source / "block.csv"),
                    "--license-file", str(self.source / "LICENSE"),
                    "--out", str(self.outside / "prepared")]
        return [command, "--primary-run", "runs/primary", "--data-cache", str(self.outside / "prepared"),
                "--out", str(self.outside / "external")]

    def test_synthetic_revision_runs_inside_outer_repository_remain_allowed(self):
        args = ["--all", "--dry-run", "--out", "runs/synthetic"]
        self.assertEqual(validate_arguments("revision", args, self.repo), args)
        self.assertFalse((self.repo / "runs").exists())

    def test_real_output_outside_inner_but_inside_outer_repository_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "entire public repository"):
            validate_arguments("revision", self.real_args(self.repo / "private_out"), self.repo)

    def test_real_input_inside_outer_repository_is_rejected(self):
        args = self.real_args(self.outside)
        args[args.index("--data-root") + 1] = str(self.repo / "data")
        with self.assertRaisesRegex(ValueError, "--data-root"):
            validate_arguments("revision", args, self.repo)

    def test_real_args_missing_required_paths_are_explicit_errors(self):
        for args in (["--real-cases"], ["--all", "--include-real", "--data-root", str(self.source)]):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "requires explicit paths"):
                validate_arguments("revision", args, self.repo)

    def test_legal_external_absolute_real_paths_are_normalized_without_creation(self):
        result = validate_arguments("review", self.real_args(self.outside / "new"), self.repo)
        self.assertEqual(option(result, "--private-out"), str(self.outside / "new"))
        self.assertFalse(self.outside.exists())
        self.assertFalse(self.source.exists())

    def test_relative_real_paths_resolve_from_repository_cwd(self):
        result = validate_arguments("revision", ["--real-cases", "--data-root", "../source", "--private-out", "../private/results"], self.repo)
        self.assertEqual(option(result, "--data-root"), str(self.source))
        self.assertEqual(option(result, "--private-out"), str(self.outside / "results"))
        with self.assertRaises(ValueError):
            validate_arguments("revision", ["--private-out", "private_out"], self.repo)

    def test_symlink_into_outer_repository_is_rejected(self):
        alias = self.base / "apparently_external"
        try:
            alias.symlink_to(self.repo, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlink fixture unavailable: {error}")
        with self.assertRaises(ValueError):
            validate_arguments("revision", self.real_args(alias / "private_out"), self.repo)

    def test_symlink_to_external_path_is_canonicalized(self):
        self.outside.mkdir()
        alias = self.repo / "external_link"
        try:
            alias.symlink_to(self.outside, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlink fixture unavailable: {error}")
        result = validate_arguments("revision", self.real_args(alias / "result"), self.repo)
        self.assertEqual(option(result, "--private-out"), str(self.outside / "result"))

    def test_ancestor_directory_cannot_contain_repository(self):
        with self.assertRaises(ValueError):
            validate_arguments("revision", self.real_args(self.base), self.repo)

    def test_help_passes_through_without_filesystem_or_data_side_effects(self):
        nonexistent = self.base / "missing_repository"
        for route, args in [("revision", ["--real-cases", "--help"]), ("external", ["prepare", "--help"]), ("legacy", ["-h"])]:
            with self.subTest(route=route):
                self.assertEqual(validate_arguments(route, args, nonexistent), args)
        self.assertFalse(nonexistent.exists())

    def test_external_prepare_uses_relocated_engine_by_default(self):
        args = self.external_args()
        original = list(args)
        result = validate_arguments("external", args, self.repo)
        self.assertEqual(option(result, "--code-root"), str(self.repo / "revision"))
        self.assertEqual(args, original)
        self.assertFalse(self.outside.exists())

    def test_explicit_engine_root_must_be_relocated_revision(self):
        args = self.external_args() + ["--code-root", "revision"]
        result = validate_arguments("porphyry", args, self.repo)
        self.assertEqual(option(result, "--code-root"), str(self.repo / "revision"))
        for wrong in [".", "legacy", str(self.base / "other_engine")]:
            with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, "--code-root"):
                validate_arguments("external", self.external_args() + ["--code-root", wrong], self.repo)

    def test_engine_symlink_cannot_redirect_to_foreign_code(self):
        (self.repo / "revision").rmdir()
        target = self.base / "foreign_engine"
        target.mkdir()
        try:
            (self.repo / "revision").symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlink fixture unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "engine must remain inside"):
            validate_arguments("external", self.external_args(), self.repo)

    def test_external_source_license_cache_and_outputs_must_leave_outer_repository(self):
        for flag in ["--source", "--license-file", "--out"]:
            args = self.external_args()
            args[args.index(flag) + 1] = str(self.repo / "outer_not_revision" / "file")
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "entire public repository"):
                validate_arguments("external", args, self.repo)
        args = self.external_args("inspect")
        args[args.index("--data-cache") + 1] = str(self.repo / "cache")
        with self.assertRaisesRegex(ValueError, "--data-cache"):
            validate_arguments("external", args, self.repo)

    def test_pure_synthetic_primary_run_may_remain_inside_repository(self):
        result = validate_arguments("external", self.external_args("run"), self.repo)
        self.assertEqual(option(result, "--primary-run"), str(self.repo / "runs/primary"))

    def test_run_and_saved_verify_require_explicit_separate_output(self):
        for command in ["run", "verify", "verify-only"]:
            args = self.external_args(command)[:-2]
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, "--out"):
                validate_arguments("external", args, self.repo)

    def test_inspect_may_omit_output(self):
        result = validate_arguments("external", self.external_args("inspect")[:-2], self.repo)
        self.assertNotIn("--out", result)

    def test_author_route_provenance_paths_are_guarded(self):
        args = self.external_args("verify-reference") + ["--author-bundle", str(self.outside / "bundle.json"),
                "--reference-run", str(self.outside / "reference"), "--external-manifest", str(self.outside / "manifest.json")]
        validate_arguments("external", args, self.repo)
        for flag in ["--author-bundle", "--reference-run", "--external-manifest"]:
            wrong = list(args)
            wrong[wrong.index(flag) + 1] = str(self.repo / "private_proof")
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                validate_arguments("external", wrong, self.repo)

    def test_integration_registration_is_required_and_guarded(self):
        args = self.external_args("integration-inspect") + ["--author-bundle", str(self.outside / "bundle.json")]
        with self.assertRaisesRegex(ValueError, "--registration"):
            validate_arguments("external", args, self.repo)
        with self.assertRaises(ValueError):
            validate_arguments("external", args + ["--registration", str(self.repo / "registration.json")], self.repo)
        validate_arguments("external", args + ["--registration", str(self.outside / "registration.json")], self.repo)

    def test_inline_values_normalize_and_protected_duplicates_fail(self):
        result = validate_arguments("revision", ["--private-out=../private/new"], self.repo)
        self.assertEqual(result, ["--private-out", str(self.outside / "new")])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_arguments("revision", ["--private-out", "../a", "--private-out=../b"], self.repo)

    def test_missing_empty_and_abbreviated_protected_values_fail(self):
        for args in [["--private-out"], ["--private-out="], ["--private-out", "--all"],
                     ["--private-o", "../private"], ["--data-r=../source"], ["--include-r"]]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_arguments("revision", args, self.repo)
        with self.assertRaisesRegex(ValueError, "abbreviations"):
            validate_arguments("external", self.external_args("inspect")[:-2] + ["--ou", "cache"], self.repo)

    def test_legacy_explicit_mesh_and_output_paths_are_guarded(self):
        for flag in ["--ply-dir", "--ply-file", "--output-dir"]:
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                validate_arguments("legacy", [flag, "repo_local"], self.repo)
        result = validate_arguments("legacy", ["--ply-dir", "../source", "--output-dir", "../private"], self.repo)
        self.assertEqual(option(result, "--output-dir"), str(self.outside))

    def test_invalid_route_command_or_argument_shape_fails_explicitly(self):
        for route, args in [("unknown", []), ("external", []), ("external", ["unknown"]),
                            ("revision", "--all"), ("revision", None), ("revision", [None]), ("revision", ["--", "--private-out", "x"])]:
            with self.subTest(route=route, args=args), self.assertRaises(ValueError):
                validate_arguments(route, args, self.repo)


if __name__ == "__main__":
    unittest.main()
