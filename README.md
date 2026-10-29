# GTLS (GPU Transit Least Squares)
A GPU algorithm for speeding up periodic transit detection based on [TLS](https://github.com/hippke/tls).

## Installation
### Requirements
A CUDA-capable GPU is required. 

### Install
This program is based on cupy. Due to there are many versions of cupy, we cannot specify the version of cupy in the package.
So, you need to install cupy manually first.
 
Please refer to [the official document](https://docs.cupy.dev/en/stable/install.html#installing-cupy), and install the version of cupy that is suitable for your environment.

After installing cupy, you can install this program by running the following command:
```bash
pip3 install gputls
```

## Usage
```python
#Assume that you have a time series data: time, flux
from gtls import gtls
model = gtls(time, flux)
period, duration, depth, T0, SDE = model.power()
```

You can also use the test script in the repository:
```bash
python3 generatedTest.py
```

### Local debugging
With your virtual environment activated and the runtime dependencies installed, run:
```bash
./debug.sh
```
This builds a wheel with `python3 -m pip wheel` (no separate `build` package
needed), reinstalls only the local package without changing its dependencies,
and runs `KeplerLongCurveSingleTest.py`. A failed build stops the script before
replacing the installed package.

The Kepler test also needs `matplotlib`, `pandas`, `astropy`,
`transitleastsquares`, `lightkurve`, and `wotan`. Its single-curve test data is
stored in `test_data/public_Q3_long_5/` and is read relative to the script
location, so it does not depend on `/mnt/HDD0/Kepler/lightcurves` or other
files referenced by `allKoi2lcResult_update.csv`. It reads FITS data directly
and uses `gputls.helpers.cleaned_array`; no external
`GTLSTest/precision/lcFuns.py` is required. Flux and its uncertainty are divided
by the same mean flux before detrending. The test currently selects GPU device 1.

For now, There are no detailed documents. Please refer to the [TLS](https://github.com/hippke/tls) first, since the usage of this program is almost the same as TLS.

## License
The GTLS(GPU Transit Least Squares) algorithm is adapted from the TLS(Transit Least Squares) algorithm by Michael Hippke & René Heller (2019).

The TLS is an open source software with MIT license. The copyright of the TLS algorithm is held by its authors.

The GTLS is also an open source software with MIT license.