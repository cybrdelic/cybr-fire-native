"""Conservative empty-brick bounds for direct trilinear volume integration.

An entry is zero only when all optical texels in the brick and its interpolation
halo are zero. No epsilon, tail deletion, simulation culling, or density edits.
"""
from pathlib import Path
import numpy as np
from .gpu import bind_texture


class OpticalBricks:
    def __init__(self, context, shape, brick_size=8):
        self.context = context
        self.shape = tuple(shape)
        self.brick_size = int(brick_size)
        self.dimensions = tuple((n + self.brick_size - 1) // self.brick_size for n in shape)
        self.bounds = context.buffer(int(np.prod(self.dimensions)))
        self.kernel = context.program(Path(__file__).resolve().parents[1] / 'shaders/bricks.comp')

    def update(self, optical_texture):
        bind_texture(self.context, optical_texture, 0)
        self.kernel.run(self.dimensions, [self.bounds], dims=self.shape,
                        brickDims=self.dimensions, brickSize=self.brick_size)

    def uniforms(self, enabled=True, cutoff=.002):
        return dict(brickDims=self.dimensions, brickSize=self.brick_size,
                    emptySpaceSkipping=int(enabled), transmittanceCutoff=float(cutoff))

    def statistics(self):
        bounds = self.bounds.read()
        return dict(brick_size=self.brick_size, brick_count=int(bounds.size),
                    active_bricks=int(np.count_nonzero(bounds)),
                    exact_empty_fraction=float(np.mean(bounds == 0)),
                    threshold=0.0, interpolation_halo_texels=1)

    def close(self):
        self.bounds.close()
