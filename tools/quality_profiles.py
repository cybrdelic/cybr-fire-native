"""CYBR FIRE quality invariants.

The native production tier is the *original* V8 discretization. Preview is
explicitly opt-in and must never be described as native V8 or 4K simulation.
This module changes numerical/encoding defaults only; it does not establish
that any particular shot is photorealistic or complete.
"""
from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class QualityProfile:
    name: str
    parent: tuple[int, int, int]
    fine: tuple[int, int, int]
    video: tuple[int, int]
    samples: int
    flow_scale: int
    wavelengths: tuple[float, float, float]

    @property
    def parent_cells(self) -> int:
        return prod(self.parent)

    @property
    def fine_cells(self) -> int:
        return prod(self.fine)


PROFILES = {
    'native-v8': QualityProfile('native-v8', (160, 280, 160), (384, 672, 384),
                                (2160, 3840), 2, 2, (.16, .08, .04)),
    'preview': QualityProfile('preview', (64, 112, 64), (144, 252, 144),
                               (720, 1280), 2, 1, (.20, .12, .08)),
}


def get_profile(name: str) -> QualityProfile:
    if name not in PROFILES:
        raise ValueError(f'Unknown quality profile: {name!r}')
    return PROFILES[name]


def require_native(parent, fine, video):
    expected = PROFILES['native-v8']
    got = (tuple(parent), tuple(fine), tuple(video))
    want = (expected.parent, expected.fine, expected.video)
    if got[:2] != want[:2] or got[2] not in (want[2],want[2][::-1]):
        raise ValueError(f'Native V8 quality regression: expected {want} or 4K landscape, got {got}')
