# GTLS (GPU Transit Least Squares)
A GPU algorithm for speeding up periodic transit detection based on [TLS](https://github.com/hippke/tls).

## Installation
### Requirements
A CUDA-capable GPU is required. For now, the GRAM of GPU should be 12GB or more. For the future, we will try our best to reduce the memory usage.

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
python3 gtlsTest.py
```

For now, There are no detailed documents. Please refer to the [TLS](https://github.com/hippke/tls) first, since the usage of this program is almost the same as TLS.

## License
The GTLS(GPU Transit Least Squares) algorithm is adapted from the TLS(Transit Least Squares) algorithm by Michael Hippke & René Heller (2019).

The TLS is an open source software with MIT license. The copyright of the TLS algorithm is held by its authors.

The GTLS is also an open source software with MIT license.