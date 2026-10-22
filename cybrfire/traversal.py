"""Measurement-driven selection between mathematically equivalent traversals.

The selector is independent of scene names, flame appearance, and adapter brands.
It may change execution work, never optical density or the sampling lattice.
"""
import math
import statistics


class TraversalSelector:
    def __init__(self, occupancy_change=.2, required_gain=1.05):
        self.occupancy_change=occupancy_change
        self.required_gain=required_gain
        self.empty_fraction=None
        self.selected=False
        self.measurements=[]

    def needs_measurement(self, empty_fraction):
        return self.empty_fraction is None or abs(empty_fraction-self.empty_fraction)>=self.occupancy_change

    def choose(self, empty_fraction, measure):
        if not self.needs_measurement(empty_fraction):return self.selected
        # Warm both execution branches, then alternate order to reduce bias.
        measure(False);measure(True)
        samples={False:[],True:[]}
        for repeat in range(3):
            for enabled in ([False,True] if repeat%2==0 else [True,False]):
                value=float(measure(enabled))
                if not math.isfinite(value) or value<=0:raise ValueError('Invalid traversal timing')
                samples[enabled].append(value)
        dense=statistics.median(samples[False]);brick=statistics.median(samples[True])
        self.selected=brick*self.required_gain<dense
        self.empty_fraction=float(empty_fraction)
        self.measurements.append(dict(exact_empty_fraction=self.empty_fraction,dense_ms=dense,
            brick_ms=brick,selected='bricks' if self.selected else 'dense',required_gain=self.required_gain,
            samples_ms={'dense':samples[False],'bricks':samples[True]},
            scope='Small camera-matched primary-ray calibration; synchronized dispatch, no readback.'))
        return self.selected
