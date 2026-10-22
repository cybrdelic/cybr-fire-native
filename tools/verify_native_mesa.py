"""Bounded real native Mesa execution, not a production or macOS guarantee."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cybrfire.config import Domain, example
from cybrfire.gpu import ComputeContext
from cybrfire.image_fields import require_native_field_bindings
from cybrfire.render import VolumeRenderer, tone_map
from cybrfire.solver import Simulation


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/native-mesa"))
    parser.add_argument("--require-platform", choices=("darwin", "linux", "win32"))
    parser.add_argument("--pressure", choices=("multigrid", "spectral"), default="multigrid")
    args = parser.parse_args()
    if args.require_platform:
        require(sys.platform == args.require_platform, "This verifier must run on the requested native host platform.")
    started = time.perf_counter()
    ctx = ComputeContext()
    renderers = []
    try:
        print(json.dumps({"host_platform": sys.platform, "adapter": ctx.info}), flush=True)
        require(ctx.info["backend"] == "egl" and ctx.info["egl_platform"] == "surfaceless",
                "Require CYBR_GL_BACKEND=egl and CYBR_EGL_PLATFORM=surfaceless.")
        require("llvmpipe" in ctx.info["renderer"].lower() and ctx.info["software_adapter"],
                "Require actual llvmpipe: set LIBGL_ALWAYS_SOFTWARE=1 and GALLIUM_DRIVER=llvmpipe before launch; check for a conflicting MESA_LOADER_DRIVER_OVERRIDE.")
        require_native_field_bindings(ctx)
        base = example("burner", "test")
        scene = replace(base, domain=Domain((16, 24, 16), .1, pressure_backend=args.pressure),
                        render=replace(base.render, width=64, height=96, samples=1,
                                       ground_resolution=16, ground_directions=16))
        legacy = Simulation(ctx, scene)
        native = Simulation(ctx, replace(scene, domain=replace(scene.domain, native_fields=True)))
        initial = native.q.read()
        legacy.step(4)
        native.step(4)
        print("Completed four real integration ticks for each storage mode.", flush=True)
        left, right = legacy.snapshot(), native.snapshot()
        require(not np.array_equal(initial, right["q"]), "Native volume did not evolve.")
        require(left["tick"] == right["tick"] == 4, "Unexpected integration tick.")
        errors = {}
        for name in ("q", "velocity", "pressure"):
            require(np.isfinite(left[name]).all() and np.isfinite(right[name]).all(), "Nonfinite " + name)
            errors[name] = float(np.max(np.abs(left[name] - right[name])))
            require(np.array_equal(left[name], right[name]), "Native/SSBO mismatch in " + name)
        slab = native.q.read_slab(1, 3)
        require(np.array_equal(slab, right["q"].reshape(16, 24, 16, 4)[1:4]), "Native slab readback mismatch.")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "native-checkpoint.npz"
            native.save(checkpoint)
            native.step(1)
            native.load(checkpoint)
            restored = native.snapshot()
            require(restored["tick"] == right["tick"], "Checkpoint tick mismatch.")
            for name in ("q", "velocity", "pressure"):
                require(np.array_equal(restored[name], right[name]), "Checkpoint mismatch in " + name)
        images = []
        for simulation in (legacy, native):
            renderer = VolumeRenderer(ctx, simulation)
            renderers.append(renderer)
            hdr = renderer.render_hdr(width=64, height=96, samples=1)
            require(np.isfinite(hdr).all() and np.ptp(hdr) > 0, "Nonfinite or constant HDR render.")
            images.append(hdr)
        errors["hdr"] = float(np.max(np.abs(images[0] - images[1])))
        require(np.array_equal(images[0], images[1]), "Native/SSBO HDR render mismatch.")
        ctx.check()
        report = {
            "passed": True, "host": {"platform": sys.platform, "machine": platform.machine()},
            "adapter": ctx.info, "scope": "four ticks, 16x24x16, native-image/SSBO parity, 64x96 render",
            "production_quality": False, "pressure_backend": args.pressure,
            "physics_scene": scene.to_dict(), "native_storage_scene": native.scene.to_dict(),
            "maximum_absolute_errors": errors, "slab_readback": True, "checkpoint_roundtrip": True,
            "diagnostics": native.diagnostics(), "wall_seconds": time.perf_counter() - started,
            "shader_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "shaders").iterdir()) if p.is_file()},
        }
        args.out.mkdir(parents=True, exist_ok=True)
        rgb = tone_map(images[1], scene.render.exposure, scene.render.bloom)
        Image.fromarray(rgb).save(args.out / "native-mesa-smoke.png")
        (args.out / "native-mesa-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        for renderer in renderers:
            renderer.close()
        ctx.close()


if __name__ == "__main__":
    main()
