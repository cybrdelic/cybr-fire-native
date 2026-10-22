"""Bounded genuine solver/render smoke; never a native production benchmark."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cybrfire.config import Domain, example
from cybrfire.gpu import ComputeContext
from cybrfire.render import VolumeRenderer, tone_map
from cybrfire.solver import Simulation

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/portable-smoke"))
    args = parser.parse_args()
    ctx = ComputeContext()
    started = time.perf_counter()
    print(json.dumps({"adapter": ctx.info, "scope": "small-grid numerical smoke"}), flush=True)
    try:
        scene = example("burner", "test")
        scene = replace(scene, domain=Domain((16,24,16), .1, pressure_backend="spectral"),
                        render=replace(scene.render, width=64, height=96, samples=1,
                                       ground_resolution=16, ground_directions=16))
        simulation = Simulation(ctx, scene)
        before = simulation.q.read()
        simulation.step(4)
        state = simulation.snapshot()
        diagnostics = simulation.diagnostics()
        assert diagnostics["finite"] and diagnostics["tick"] == 4
        for key in ("q", "velocity", "pressure"):
            assert np.isfinite(state[key]).all(), key + " contains nonfinite values"
        assert not np.array_equal(before, state["q"]), "volume did not evolve"
        renderer = VolumeRenderer(ctx, simulation)
        try:
            hdr = renderer.render_hdr(width=64, height=96, samples=1)
            assert np.isfinite(hdr).all(), "nonfinite radiance"
            rgb = tone_map(hdr, renderer.cfg.exposure, renderer.cfg.bloom)
            assert np.ptp(rgb) > 0, "constant output image"
            ctx.check()
            args.out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(args.out / "small-grid-smoke.png")
            report = {"passed": True, "scope": "four ticks, 16x24x16 grid, 64x96 render",
                      "production_quality": False, "scene": scene.to_dict(),
                      "adapter": ctx.info, "diagnostics": diagnostics,
                      "finite_radiance": True, "wall_seconds": time.perf_counter()-started}
            (args.out / "smoke.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2), flush=True)
        finally:
            renderer.close()
    finally:
        ctx.close()

if __name__ == "__main__":
    main()
