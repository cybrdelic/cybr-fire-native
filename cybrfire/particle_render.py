"""Demand-sized particle workspaces; no particles are discarded on overflow.

Tile counts include every projected candidate even when the initial index buffer
is too small. The owner grows the workspace and repeats binning before exposing
it. Per-sample event demand is handled by the volume integrator similarly.
Neither policy changes simulation state, optical coefficients, or visibility.
"""
from pathlib import Path
import numpy as np


def grown_capacity(required: int, maximum: int) -> int:
    """Geometric growth bounded by the number of possible particle candidates."""
    if required < 1 or required > maximum:
        raise RuntimeError('Particle workspace demand exceeds the live-state capacity')
    return min(maximum, 1 << (required - 1).bit_length())


class ParticleRenderer:
    def __init__(self, ctx, sim, lut):
        self.ctx, self.sim, self.lut = ctx, sim, lut
        self.shape = None
        self.projected = ctx.buffer((sim.scene.embers.capacity, 2, 4))
        self.counts = None
        self.indices = None
        self.overflow = ctx.buffer(1, dtype=np.uint32)
        self.kernel = ctx.program(Path(__file__).resolve().parents[1] / 'shaders/particle_bins.comp')
        self.tile_capacity_high_water = 0
        self.tile_growths = 0
        self.max_tile_occupancy = 0

    def _allocate(self, tx, ty, capacity):
        shape = (tx, ty, capacity)
        if shape == self.shape:
            return
        # Replace only after both allocations succeed. A failure cannot leave
        # the owner pointing at released buffers from the preceding frame.
        counts = self.ctx.buffer(tx * ty + 1, dtype=np.uint32)
        indices = None
        try:
            indices = self.ctx.buffer(tx * ty * capacity, dtype=np.uint32)
            self.ctx.check()
        except Exception:
            counts.close()
            if indices is not None:
                indices.close()
            raise
        old_counts, old_indices = self.counts, self.indices
        self.counts, self.indices, self.shape = counts, indices, shape
        if old_counts is not None:
            old_counts.close()
            old_indices.close()

    def prepare(self, w, h, camera, cfg):
        tx, ty = (w + 15) // 16, (h + 15) // 16
        maximum = self.sim.scene.embers.capacity
        capacity = min(maximum, max(cfg.ember_tile_capacity, self.tile_capacity_high_water))
        while True:
            self._allocate(tx, ty, capacity)
            kw = dict(resolution=(w, h), tileDims=(tx, ty), tileCapacity=capacity,
                      particleCount=maximum, emissivity=self.sim.scene.embers.emissivity, **camera)
            bindings = [self.sim.embers.state, self.projected, self.counts, self.indices, self.lut]
            self.kernel.run(tx * ty + 1, bindings, operation=0, **kw)
            self.kernel.run(maximum, bindings, operation=1, **kw)
            values = self.counts.read()
            required = int(values[:-1].max(initial=0))
            self.max_tile_occupancy = required
            if required <= capacity and not int(values[-1]):
                break
            if required <= capacity:
                raise RuntimeError('Inconsistent particle bin overflow accounting')
            capacity = grown_capacity(required, maximum)
            self.tile_growths += 1
        self.tile_capacity_high_water = max(self.tile_capacity_high_water, capacity)
        self.overflow.zero()
        self.sim.solids.bind()

    def uniforms(self):
        return dict(tileDims=self.shape[:2], tileCapacity=self.shape[2])

    def event_demand(self):
        return int(self.overflow.read()[0])

    def check(self):
        if self.event_demand():
            raise RuntimeError('Particle event workspace must be retried before publishing this image')

    def close(self):
        self.projected.close()
        self.overflow.close()
        if self.counts is not None:
            self.counts.close()
            self.indices.close()
