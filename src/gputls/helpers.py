import numpy
from numpy import arccos, degrees


def cleaned_array(t, y, dy=None):
    """Takes numpy arrays with masks and non-float values.
    Returns unmasked cleaned arrays."""

    def isvalid(value):
        valid = False
        if value is not None:
            if not numpy.isnan(value):
                if value > 0 and value < numpy.inf:
                    valid = True
        return valid

    # Start with empty Python lists and convert to numpy arrays later (reason: speed)
    clean_t = []
    clean_y = []
    if dy is not None:
        clean_dy = []

    # Cleaning numpy arrays with both NaN and None values is not trivial, as the usual
    # mask/delete filters do not accept their simultanous ocurrence without warnings.
    # Instead, we iterate over the array once; this is not Pythonic but works reliably.
    for i in range(len(y)):

        # Case: t, y, dy
        if dy is not None:
            if isvalid(y[i]) and isvalid(t[i]) and isvalid(dy[i]):
                clean_y.append(y[i])
                clean_t.append(t[i])
                clean_dy.append(dy[i])

        # Case: only t, y
        else:
            if isvalid(y[i]) and isvalid(t[i]):
                clean_y.append(y[i])
                clean_t.append(t[i])

    clean_t = numpy.array(clean_t, dtype=float)
    clean_y = numpy.array(clean_y, dtype=float)

    if dy is None:
        return clean_t, clean_y
    else:
        clean_dy = numpy.array(clean_dy, dtype=float)
        return clean_t, clean_y, clean_dy


def transit_mask(t, period, duration, T0):
    # Works with numba, but is not faster
    mask = numpy.abs((t - T0 + 0.5 * period) % period - 0.5 * period) < 0.5 * duration
    return mask


def running_median(data, kernel):
    """Returns sliding median of width 'kernel' and same length as data """
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
