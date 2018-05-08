# pycochleagram readme
Generate cochleagrams natively in Python. Ported from Josh McDermott's MATLAB code. 

<!-- ![Cochleagram Icon](docs/source/cochleagramIcon.png =890x260) -->

## Documentation
The documentation can be found on [Read the Docs](https://pycochleagram.readthedocs.io/en/latest/readmeLink.html), the source is on [github](https://github.com/mcdermottLab/pycochleagram). 

## Installation
**Note**: Installation via pip is planned, but not currently supported.

1) Download or clone this git repository containing the pycochleagram source code. 
```
git clone https://github.com/mcdermottLab/pycochleagram
```
2) Navigate to the downloaded pycochleagram folder. 
```
cd pycochleagram
```
3) Build and install using setup.py. 
```
python setup.py install
```

Imports should work like `import pycochleagram.cochleagram as cgram`. You should be able to test the installation by running the [demo](#demo).

## Demo
There are a few demos in `demo.py`. To run these demos, navigate to the root of the project folder and run `python demo.py`. Note, you can enable audio playback with `python demo.py -p`, but be warned that this can get **very loud**. 

## Overview
**Note**: Numpy/Scipy's FFT implementation ([fftpack](https://docs.scipy.org/doc/scipy/reference/fftpack.html)) can be ridiculously slow for certain sized inputs. You can install [pyfftw](https://github.com/pyFFTW/pyFFTW) to get around this issue, but this can be painful on computers without install install privileges (shared servers, clusters, etc. -- consider using a software container in such cases).

This package contains four main modules:
+ [**pycochleagram.erbfilter**](https://pycochleagram.readthedocs.io/en/latest/pycochleagram.html#module-pycochleagram.erbfilter):
Functions for generating ERB-cosine filters. These functions
are available in the original MATLAB implementation. 

+ [**pycochleagram.subband**](https://pycochleagram.readthedocs.io/en/latest/pycochleagram.html#module-pycochleagram.subband):
Functions for performing subband decomposition using filterbanks made with `erbfilter`. These functions are available in the MATLAB implementation. 

+ [**pycochleagram.cochleagram**](https://pycochleagram.readthedocs.io/en/latest/pycochleagram.html#module-pycochleagram.cochleagram):
Convenience methods for quickly generating cochleagrams. Also, provides functions for cochleagram inversion (i.e., generating a signal waveform from a cochleagram). These methods are not readily available in the MATLAB implementation (you would have to compose functions from pycochleagram.erbfilter and pycochleagram.subband). This is intended to help you get started.

+ [**pycochleagram.utils**](https://pycochleagram.readthedocs.io/en/latest/pycochleagram.html#module-pycochleagram.utils):
A collection of helpful methods for working with cochleagram generation, including some plotting and audio playback functions, as well as some fft-like methods that allow for easy switching between fftpack (numpy/scipy) and fftw. 
**NOTE**: when working with pyaudio and the audio playback functions in `utils`, the sound output can be **very loud**. Take caution when working with this method.

## TODO:
+ convert docstrings to google format
+ write readme
+ build and format docs
+ clean up old and deprecated methods
+ dependencies
+ cochleagram description
+ hack for polyphase resampling error
+ sensible parameters for downsampling?
+ python compatibility issues
+ test padding (pad_factor)
+ erb filters fails with certain arguments: `N: 680, sample_factor: 15, signal_length: 2433, sr: 32593, low_lim: 147, hi_lim: 16296, pad_factor: None`
