"""CPU-only portability contracts, runnable on Windows, Linux and macOS."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cybrfire.file_lock import lock_exclusive
from cybrfire.gl_context import open_context
from cybrfire.doctor import diagnose

class PortableContracts(unittest.TestCase):
    def test_lock_excludes_another_process_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.lock"
            code = ("import sys; from cybrfire.file_lock import lock_exclusive; "
                    "f=open(sys.argv[1], 'a+'); lock_exclusive(f)")
            with path.open("a+") as held:
                lock_exclusive(held)
                blocked = subprocess.run([sys.executable, "-c", code, str(path)],
                                         cwd=ROOT, capture_output=True, timeout=30)
                self.assertNotEqual(blocked.returncode, 0)
                self.assertIn(b"BlockingIOError", blocked.stderr)
            released = subprocess.run([sys.executable, "-c", code, str(path)],
                                      cwd=ROOT, capture_output=True, timeout=30)
            self.assertEqual(released.returncode, 0, released.stderr.decode())

    def test_nested_checkpoint_paths_roundtrip_and_read_old_windows_manifests(self):
        import importlib.util
        import json
        spec = importlib.util.spec_from_file_location("guard_under_test", ROOT / "checkpoint_guard/checkpoint_guard.py")
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run = base / "run"
            (run / "parent").mkdir(parents=True)
            (run / "parent/state.bin").write_bytes(b"nested numerical checkpoint")
            store = base / "store"
            manifest = guard.snapshot(run, store, ["parent/state.bin"], "nested")
            self.assertEqual(manifest["files"][0]["path"], "parent/state.bin")
            guard.restore(store, base / "restored")
            self.assertEqual((base / "restored/parent/state.bin").read_bytes(), b"nested numerical checkpoint")
            manifest["files"][0]["path"] = "parent" + chr(92) + "state.bin"
            (store / "generations/nested/manifest.json").write_text(json.dumps(manifest))
            guard.restore(store, base / "legacy-restored")
            self.assertEqual((base / "legacy-restored/parent/state.bin").read_bytes(), b"nested numerical checkpoint")
        for unsafe in ("../escape", ".." + chr(92) + "escape", "C:/escape", "C:" + chr(92) + "escape"):
            with self.assertRaises(ValueError):
                guard.protected_rel(unsafe)

    def test_native_binding_reservation_still_rejects_eight_unit_adapters(self):
        from cybrfire.image_fields import require_native_field_bindings
        class EightUnitContext:
            def glGetIntegerv(self, enum, output):
                output._obj.value = 8
        with self.assertRaisesRegex(RuntimeError, "require 12; adapter exposes 8"):
            require_native_field_bindings(EightUnitContext())

    def test_invalid_backend_is_rejected_before_context_creation(self):
        with patch.dict(os.environ, {"CYBR_GL_BACKEND": "not-a-backend"}):
            with self.assertRaisesRegex(ValueError, "CYBR_GL_BACKEND"):
                open_context()

    def test_capability_report_does_not_claim_unprobed_compute(self):
        report = diagnose(probe=False)
        self.assertIsNone(report["compute_supported"])
        self.assertIsNone(report["adapter"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
