import cupy
import numpy
from cupyx.scipy.ndimage import median_filter
from numpy import arccos, degrees


def cleaned_array(t, y, dy=None):
    """Takes numpy arrays with masks and non-float values.
    Returns unmasked cleaned arrays."""

    def as_float(values):
        # None and masked entries become NaN
        return numpy.ma.filled(numpy.ma.asarray(values).astype(float), numpy.nan)

    arrays = [as_float(a) for a in ((t, y) if dy is None else (t, y, dy))]
    valid = numpy.ones(len(arrays[1]), dtype=bool)
    for values in arrays:
        with numpy.errstate(invalid="ignore"):
            valid &= numpy.isfinite(values) & (values > 0)
    return tuple(values[valid] for values in arrays)


def transit_mask(t, period, duration, T0):
    # Works with numba, but is not faster
    mask = numpy.abs((t - T0 + 0.5 * period) % period - 0.5 * period) < 0.5 * duration
    return mask


def running_median(data, kernel):
    """Returns sliding median of width 'kernel' and same length as data """
    width = len(numpy.arange(kernel))  # kernel may be a float
    if not numpy.ma.is_masked(data) and width % 2 == 1 and len(data) >= width:
        # Same medians as the sliding windows below, computed on the GPU
        med = median_filter(cupy.asarray(numpy.ma.getdata(data)), size=width, mode="nearest").get()
        med = med[width // 2:len(data) - width // 2]
    else:
        idx = numpy.arange(kernel) + numpy.arange(len(data) - kernel + 1)[:, None]
        idx = idx.astype(int)  # needed if oversampling_factor is not int
        med = numpy.ma.median(data[idx], axis=1)

    # Append the first/last value at the beginning/end to match the length of
    # data and returned median
    first_values = med[0]
    last_values = med[-1]
    missing_values = len(data) - len(med)
    values_front = int(missing_values * 0.5)
    values_end = missing_values - values_front
    med = numpy.append(numpy.full(values_front, first_values), med)
    med = numpy.append(med, numpy.full(values_end, last_values))
    return med


def impact_to_inclination(b, semimajor_axis):
    """Converts planet impact parameter b = [0..1.x] to inclination [deg]"""
    return degrees(arccos(b / semimajor_axis))
