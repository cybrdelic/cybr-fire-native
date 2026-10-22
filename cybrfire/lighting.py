"""Positive angular quadrature and bounded source iteration for volume lighting.

This is a low-resolution isotropic-scattering approximation, not a path tracer.
The 14-node quadrature integrates sphere constants and quadratic moments exactly.
Six-node / one-iteration mode retains the V4 angular model as an ablation.
"""
from itertools import product
from pathlib import Path
import numpy as np
from .gpu import texture3d, bind_texture, delete_texture


def quadrature(count=14):
    if count not in (6, 14):
        raise ValueError('Angular quadrature must contain 6 or 14 directions')
    axes = []
    for axis in range(3):
        for sign in (-1, 1):
            direction = [0, 0, 0]
            direction[axis] = sign
            axes.append((tuple(direction), 1 / 6 if count == 6 else 1 / 15))
    if count == 14:
        axes.extend((tuple(d), 3 / 40) for d in product((-1, 1), repeat=3))
    return tuple(axes)


class IncidentLighting:
    def __init__(self, context, simulation_shape, scale=2):
        self.context = context
        self.scale = scale
        self.shape = tuple(max(4, n // scale) for n in simulation_shape)
        self.accumulation = context.buffer((int(np.prod(self.shape)), 4))
        self.textures = [texture3d(context, self.shape, 1, full_precision=True),
                         texture3d(context, self.shape, 6, full_precision=True)]
        self.texture = self.textures[0]
        self.kernel = context.program(Path(__file__).resolve().parents[1] / 'shaders/lighting.comp')

    def prepare(self, optical_texture, spacing, config, solids=None):
        ctx = self.context
        if solids:solids.bind()
        nodes = quadrature(config.illumination_directions)
        # Preserve V4's mean clear-air incident environment across quadratures.
        # The downward direction samples a brighter overhead environment.
        brightness = [.45 + 1.55 * max(-d[1] / np.linalg.norm(d), 0.)**2 for d, _ in nodes]
        normalization = (4.25 / 6) / sum(w * b for (_, w), b in zip(nodes, brightness))
        bind_texture(ctx, optical_texture, 0)
        for iteration in range(config.scattering_iterations):
            target = self.textures[iteration % 2]
            previous = self.textures[(iteration + 1) % 2]
            bind_texture(ctx, previous, 6)
            bind_texture(ctx, target, 1, True)
            self.kernel.run(self.shape, [self.accumulation], dims=self.shape, operation=2)
            for (direction, weight), b in zip(nodes, brightness):
                if sum(abs(v) for v in direction) == 3:
                    nx, ny, nz = self.shape
                    rays = ny*nz + (nx-1)*nz + (nx-1)*(ny-1)
                else:
                    axis = next(i for i, v in enumerate(direction) if v)
                    rays = self.shape[(axis+1) % 3] * self.shape[(axis+2) % 3]
                boundary = tuple(float(config.ambient * b * normalization * c) for c in (.65, .78, 1.))
                self.kernel.run(rays, [self.accumulation], dims=self.shape, operation=0,
                    sweepDirection=direction, angularWeight=float(weight), boundaryLight=boundary,
                    coarseSpacing=float(spacing * self.scale), scatterAlbedo=float(config.scatter_albedo),
                    scatterIteration=iteration,solidCount=len(solids.solids) if solids else 0)
            self.kernel.run(self.shape, [self.accumulation], dims=self.shape, operation=1)
            self.texture = target
        bind_texture(ctx, self.texture, 1)
        return self.texture

    def close(self):
        self.accumulation.close()
        for handle in self.textures:
            delete_texture(self.context, handle)
