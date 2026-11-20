import os
import warnings

import numpy

from . import constants
from .helpers import cleaned_array, impact_to_inclination


def validate_inputs(t, y, dy):
    """Check the consistency of the inputs"""

    # Clean array
    if dy is None:
        t, y = cleaned_array(t, y)
    else:
        t, y, dy = cleaned_array(t, y, dy)
        # Normalize dy to act as weights in least squares calculatio
        dy = dy / numpy.mean(dy)

    duration = max(t) - min(t)
    if duration <= 0:
        raise ValueError("Time duration must positive")
    if numpy.size(y) < 3 or numpy.size(t) < 3:
        raise ValueError("Too few values in data set")
    if numpy.mean(y) > 1.01 or numpy.mean(y) < 0.99:
        text = (
            "Warning: The mean flux should be normalized to 1"
            + ", but it was found to be "
            + str(numpy.mean(y))
        )
        warnings.warn(text)

    if min(y) < 0:
        raise ValueError("Flux values must be positive")
    if max(y) >= float("inf"):
        raise ValueError("Flux values must be finite")

    # If no dy is given, create it with the standard deviation of the flux
    if dy is None:
        dy = numpy.full(len(y), numpy.std(y))
    if numpy.size(t) != numpy.size(y) or numpy.size(t) != numpy.size(dy):
        raise ValueError("Arrays (t, y, dy) must be of the same dimensions")
    if t.ndim != 1:  # Size identity ensures dimensional identity
        raise ValueError("Inputs (t, y, dy) must be 1-dimensional")

    return t, y, dy


def validate_args(self, kwargs):
    """Validate **kwargs and set to defaults where missing"""

    # Warn user if unknown parameters
    for key in kwargs:
        if key not in constants.VALID_PARAMETERS:
            warnings.warn("Ignoring unknown parameter: " + str(key))

    self.backend = kwargs.get("backend") or os.environ.get("TLS_BACKEND") or "fused-pl"
    if self.backend not in ("fused-pl", "fused"):
        raise ValueError(f"Unknown backend {self.backend!r}. Available: 'fused-pl', 'fused'")
    self.T0_coarsen = kwargs.get("T0_coarsen", os.environ.get("TLS_T0_COARSEN"))
    if self.T0_coarsen is None:
        self.T0_coarsen = 3 if self.backend == "fused-pl" else 1
    if str(self.T0_coarsen) not in ("1", "3"):
        raise ValueError("T0_coarsen must be 1 or 3")
    self.T0_coarsen = int(self.T0_coarsen)
    self.GPUDeviceID = kwargs.get("GPUDeviceID", constants.GPU_DEVICE_ID)
    # Multi-GPU: GPUDeviceIDs overrides GPUDeviceID; the first device does the final fit
    device_ids = kwargs.get("GPUDeviceIDs")
    if device_ids is None:
        self.GPUDeviceIDs = [self.GPUDeviceID]
    else:
        self.GPUDeviceIDs = list(dict.fromkeys(int(d) for d in numpy.atleast_1d(device_ids)))
        if len(self.GPUDeviceIDs) == 0:
            raise ValueError("GPUDeviceIDs must contain at least one device")
    self.fast = kwargs.get("fast", constants.FAST)
    self.bar_location = kwargs.get("bar_location", constants.BAR_LOCATION)

    self.transit_depth_min = kwargs.get(
        "transit_depth_min", constants.TRANSIT_DEPTH_MIN
    )
    self.R_star = kwargs.get("R_star", constants.R_STAR)
    self.M_star = kwargs.get("M_star", constants.M_STAR)
    self.oversampling_factor = kwargs.get(
        "oversampling_factor", constants.OVERSAMPLING_FACTOR
    )
    self.period_max = kwargs.get("period_max", float("inf"))
    self.period_min = kwargs.get("period_min", 0)
    self.periods = kwargs.get("periods", [])
    self.n_transits_min = kwargs.get("n_transits_min", constants.N_TRANSITS_MIN)

    self.R_star_min = kwargs.get("R_star_min", constants.R_STAR_MIN)
    self.R_star_max = kwargs.get("R_star_max", constants.R_STAR_MAX)
    self.M_star_min = kwargs.get("M_star_min", constants.M_STAR_MIN)
    self.M_star_max = kwargs.get("M_star_max", constants.M_STAR_MAX)
    self.duration_grid_step = kwargs.get(
        "duration_grid_step", constants.DURATION_GRID_STEP
    )

    self.per = kwargs.get("per", constants.DEFAULT_PERIOD)
    self.rp = kwargs.get("rp", constants.DEFAULT_RP)
    self.a = kwargs.get("a", constants.DEFAULT_A)

    self.T0_fit_margin = kwargs.get("T0_fit_margin", constants.T0_FIT_MARGIN)
    if "T0_refine_margin" in kwargs:
        warnings.warn("T0_refine_margin is deprecated. The period grid uses T0_fit_margin in a single pass.",
                      DeprecationWarning, stacklevel=3)
    self.screening_tolerances = kwargs.get("screening_tolerances")
    self.screening_policy = kwargs.get("screening_policy", "fixed")
    if self.screening_policy not in ("fixed", "auto"):
        raise ValueError("screening_policy must be 'fixed' or 'auto'")
    if self.screening_policy == "auto" and self.screening_tolerances is not None:
        raise ValueError("screening_policy='auto' cannot be combined with screening_tolerances")
    if self.screening_tolerances is not None:
        try:
            tolerances = numpy.asarray(self.screening_tolerances, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("screening_tolerances must be a decreasing sequence of positive finite values") from exc
        if (tolerances.ndim != 1 or not numpy.all(numpy.isfinite(tolerances))
                or numpy.any(tolerances <= 0) or numpy.any(numpy.diff(tolerances) >= 0)):
            raise ValueError("screening_tolerances must be a decreasing sequence of positive finite values")
        self.screening_tolerances = tuple(float(value) for value in tolerances)

    # If an impact parameter is given, it overrules the supplied inclination
    if "b" in kwargs:
        self.b = kwargs.get("b")
        self.inc = impact_to_inclination(b=self.b, semimajor_axis=self.a)
    else:
        self.inc = kwargs.get("inc", constants.DEFAULT_INC)

    self.ecc = kwargs.get("ecc", constants.DEFAULT_ECC)
    self.w = kwargs.get("w", constants.DEFAULT_W)
    self.u = kwargs.get("u", constants.DEFAULT_U)
    self.limb_dark = kwargs.get("limb_dark", constants.DEFAULT_LIMB_DARK)

    self.transit_template = kwargs.get("transit_template", "default")
    if self.transit_template == "default":
        self.per = constants.DEFAULT_PERIOD
        self.rp = constants.DEFAULT_RP
        self.a = constants.DEFAULT_A
        self.inc = constants.DEFAULT_INC

    elif self.transit_template == "grazing":
        self.b = constants.GRAZING_B
        self.inc = impact_to_inclination(b=self.b, semimajor_axis=self.a)

    elif self.transit_template == "box":
        self.per = constants.BOX_PERIOD
        self.rp = constants.BOX_RP
        self.a = constants.BOX_A
        self.b = constants.BOX_B
        self.inc = constants.BOX_INC
        self.u = constants.BOX_U
        self.limb_dark = constants.BOX_LIMB_DARK

    else:
        raise ValueError(
            'Unknown transit_template. Known values: \
            "default", "grazing", "box"'
        )

    """Validations to avoid (garbage in ==> garbage out)"""

    # Stellar radius
    # 0 < R_star < inf
    if self.R_star <= 0 or self.R_star >= float("inf"):
        raise ValueError("R_star must be positive")

    # Assert (0 < R_star_min <= R_star)
    if self.R_star_min > self.R_star:
        raise ValueError("R_star_min <= R_star is required")
    if self.R_star_min <= 0 or self.R_star_min >= float("inf"):
        raise ValueError("R_star_min must be positive")

    # Assert (R_star <= R_star_max < inf)
    if self.R_star_max < self.R_star:
        raise ValueError("R_star_max >= R_star is required")
    if self.R_star_max <= 0 or self.R_star_max >= float("inf"):
        raise ValueError("R_star_max must be positive")

    # Stellar mass
    # Assert (0 < M_star < inf)
    if self.M_star <= 0 or self.M_star >= float("inf"):
        raise ValueError("M_star must be positive")

    # Assert (0 < M_star_min <= M_star)
    if self.M_star_min > self.M_star:
        raise ValueError("M_star_min <= M_star is required")
    if self.M_star_min <= 0 or self.M_star_min >= float("inf"):
        raise ValueError("M_star_min must be positive")

    # Assert (M_star <= M_star_max < inf)
    if self.M_star_max < self.M_star:
        raise ValueError("M_star_max >= M_star required")
    if self.M_star_max <= 0 or self.M_star_max >= float("inf"):
        raise ValueError("M_star_max must be positive")

    # Period grid
    if self.period_min < 0:
        raise ValueError("period_min >= 0 required")
    if self.period_min >= self.period_max:
        raise ValueError("period_min < period_max required")
    if not isinstance(self.n_transits_min, int):
        raise ValueError("n_transits_min must be an integer value")
    if self.n_transits_min < 1:
        raise ValueError("n_transits_min must be an integer value >= 1")

    # Clip the T0 steps to [0, 0.125] (sensible limit 12.5% of transit duration)
    self.T0_fit_margin = min(max(self.T0_fit_margin, 0), 0.125)
    return self, kwargs
