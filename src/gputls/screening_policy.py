"""Per-search calibration of conservative screening cascade lengths.

Only the amount of pruning work changes. Every surviving candidate still uses
the original exact kernel; timing noise can affect speed, never eligibility.
"""
import math


class BatchScreeningPolicy:
    """Cache measurements by period band within one device/light-curve search."""

    def __init__(self, levels, points):
        self.levels = int(levels)
        self.points = int(points)
        self.band_octaves = 5 if self.points < 32768 else 3
        self.decisions = {}
        self.calibrations = []
        self.batches = []

    def group(self, periods):
        # Eightfold bands capture the steep rise of long-period template work.
        # Small curves have fewer batches: use wider bands to amortize pilots.
        middle = float(periods[len(periods) // 2])
        return math.floor(math.log2(middle) / self.band_octaves)

    @staticmethod
    def choose(timings):
        """Prefer fewer passes when they are within 5% of the fastest pilot."""
        costs = {level: min(values) for level, values in timings.items()}
        best = min(costs.values())
        return min(level for level, cost in costs.items() if cost <= best * 1.05)

    def record_batch(self, group, levels, periods, refine):
        self.batches.append(dict(group=group, levels=levels, periods=len(periods),
                                 period_min=float(min(periods)),
                                 period_max=float(max(periods)), refine=bool(refine)))

    def summary(self):
        return dict(calibration_seconds=sum(row['wall_seconds'] for row in self.calibrations),
                    band_octaves=self.band_octaves, selected_levels=self.decisions.copy(),
                    calibrations=self.calibrations.copy(), batches=self.batches.copy())

