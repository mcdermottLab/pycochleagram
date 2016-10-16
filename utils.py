from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib.pyplot import imshow, show
import numpy as np


def matlab_arange(start, stop, num):
  """Mimics MATLAB's sequence generation.

  Returns `num + 1` evenly spaced samples, calculated over the interval
  [`start`, `stop`].

  Args:
    start (scalar): The starting value of the sequence.
    stop (scalar): The end value of the sequence.
    num (int): Number of samples to generate.

  Returns:
    ndarray:
    **samples**: There are `num + 1` equally spaced samples in the closed
    interval.
  """
  return np.linspace(start, stop, num + 1)


def cochshow(cochleagram, interact=True, cmap='viridis'):
  """Helper function to facilitate displaying cochleagrams.

  Args:
    cochleagram (array): Cochleagram to display with matplotlib.
    interact (bool, optional): Determines if interactive plot should be shown.
      If True (default), plot will be shown. If this is False, the figure will
      be created but not displayed.

  Returns:
    AxesImage:
      **image**: Whatever matplotlib.pyplot.plt returns.
  """
  f = imshow(cochleagram, aspect='auto', cmap=cmap)
  if interact:
    show()
  return f


def fft(a, n=None, axis=-1, norm=None, mode='auto', params=None):
  """Provides support for various implementations of the FFT, using numpy's
  fftpack or pyfftw's fftw. This uses a numpy.fft-like interface.

  Args:
    a (array): Time-domain signal.
    mode (str): Determines which FFT implementation will be used. Options are
      'fftw', 'np', and 'auto'. Using 'auto', will attempt to use a pyfftw
      implementation with some sensible parameters (if the module is
      available), and will use numpy's fftpack implementation otherwise.
    n (int, optional): Length of the transformed axis of the output. If n is
      smaller than the length of the input, the input is cropped. If it is
      larger, the input is padded with zeros. If n is not given, the length of
      the input along the axis specified by axis is used.
    axis (int, optional): Axis over which to compute the FFT. If not given, the
      last axis is used.
    norm ({None, 'ortho'}, optional): Support for numpy interface.
    params (dict, None, optional): Dictionary of additional input arguments to
      provide to the appropriate fft function (usually fftw). Note, named
      arguments (e.g., `n`, `axis`, and `norm`) will override identically named
      arguments in `params`. If `mode` is 'auto' and `params` dict is None,
      sensible values will be chosen. If `params` is not None, it will not be
      altered.

  Returns:
    array:
      **fft_a**: Signal in the frequency domain in FFT standard order. See numpy.fft() for
      a description of the output.
  """
  # handle 'auto' mode
  mode, params = _parse_mode(mode, params)
  # named args override params
  params = {**params, 'n': n, 'axis': axis, 'norm': norm}

  if mode == 'fftw':
    import pyfftw
    return pyfftw.interfaces.numpy_fft.fft(a, **params)
  elif mode == 'np':
    return np.fft.fft(a, **params)
  else:
    raise NotImplementedError('`fft method is not defined for mode `%s`;' +
                              'use "auto", "np" or "fftw".')


def ifft(a, n=None, axis=-1, norm=None, mode='auto', params=None):
  """Provides support for various implementations of the IFFT, using numpy's
  fftpack or pyfftw's fftw. This uses a numpy.fft-like interface.

  Args:
    a (array): Time-domain signal.
    mode (str): Determines which IFFT implementation will be used. Options are
      'fftw', 'np', and 'auto'. Using 'auto', will attempt to use a pyfftw
      implementation with some sensible parameters (if the module is
      available), and will use numpy's fftpack implementation otherwise.
    n (int, optional): Length of the transformed axis of the output. If n is
      smaller than the length of the input, the input is cropped. If it is
      larger, the input is padded with zeros. If n is not given, the length of
      the input along the axis specified by axis is used.
    axis (int, optional): Axis over which to compute the FFT. If not given, the
      last axis is used.
    norm ({None, 'ortho'}, optional): Support for numpy interface.
    params (dict, None, optional): Dictionary of additional input arguments to
      provide to the appropriate fft function (usually fftw). Note, named
      arguments (e.g., `n`, `axis`, and `norm`) will override identically named
      arguments in `params`. If `mode` is 'auto' and `params` dict is None,
      sensible values will be chosen. If `params` is not None, it will not be
      altered.

  Returns:
    array:
    **ifft_a**: Signal in the time domain. See numpy.ifft() for a
      description of the output.
  """
  # handle 'auto' mode
  mode, params = _parse_mode(mode, params)
  # named args override params
  params = {**params, 'n': n, 'axis': axis, 'norm': norm}

  if mode == 'fftw':
    import pyfftw
    return pyfftw.interfaces.numpy_fft.ifft(a, **params)
  elif mode == 'np':
    return np.fft.ifft(a, **params)
  else:
    raise NotImplementedError('`ifft method is not defined for mode `%s`;' +
                              'use "np" or "fftw".')


def rfft(a, n=None, axis=-1, mode='auto', params=None):
  """Provides support for various implementations of the RFFT, using numpy's
  fftpack or pyfftw's fftw. This uses a numpy.fft-like interface.

  Args:
    a (array): Time-domain signal.
    mode (str): Determines which FFT implementation will be used. Options are
      'fftw', 'np', and 'auto'. Using 'auto', will attempt to use a pyfftw
      implementation with some sensible parameters (if the module is
      available), and will use numpy's fftpack implementation otherwise.
    n (int, optional): Length of the transformed axis of the output. If n is
      smaller than the length of the input, the input is cropped. If it is
      larger, the input is padded with zeros. If n is not given, the length of
      the input along the axis specified by axis is used.
    axis (int, optional): Axis over which to compute the FFT. If not given, the
      last axis is used.
    params (dict, None, optional): Dictionary of additional input arguments to
      provide to the appropriate fft function (usually fftw). Note, named
      arguments (e.g., `n` and `axis`) will override identically named
      arguments in `params`. If `mode` is 'auto' and `params` dict is None,
      sensible values will be chosen. If `params` is not None, it will not be
      altered.

  Returns:
    array:
    **rfft_a**: Signal in the frequency domain in standard order.
      See numpy.rfft() for a description of the output.
  """
  # handle 'auto' mode
  mode, params = _parse_mode(mode, params)
  # named args override params
  params = {**params, 'n': n, 'axis': axis}

  if mode == 'fftw':
    import pyfftw
    return pyfftw.interfaces.numpy_fft.rfft(a, **params)
  elif mode == 'np':
    return np.fft.rfft(a, **params)
  else:
    raise NotImplementedError('`rfft method is not defined for mode `%s`;' +
                              'use "np" or "fftw".')


def irfft(a, n=None, axis=-1, mode='auto', params=None):
  """Provides support for various implementations of the IRFFT, using numpy's
  fftpack or pyfftw's fftw. This uses a numpy.fft-like interface.

  Args:
    a (array): Time-domain signal.
    mode (str): Determines which FFT implementation will be used. Options are
      'fftw', 'np', and 'auto'. Using 'auto', will attempt to use a pyfftw
      implementation with some sensible parameters (if the module is
      available), and will use numpy's fftpack implementation otherwise.
    n (int, optional): Length of the transformed axis of the output. If n is
      smaller than the length of the input, the input is cropped. If it is
      larger, the input is padded with zeros. If n is not given, the length of
      the input along the axis specified by axis is used.
    axis (int, optional): Axis over which to compute the FFT. If not given, the
      last axis is used.
    params (dict, None, optional): Dictionary of additional input arguments to
      provide to the appropriate fft function (usually fftw). Note, named
      arguments (e.g., `n` and `axis`) will override identically named
      arguments in `params`. If `mode` is 'auto' and `params` dict is None,
      sensible values will be chosen. If `params` is not None, it will not be
      altered.

  Returns:
    array:
    **irfft_a**: Signal in the time domain. See numpy.irfft() for a
      description of the output.
  """
  # handle 'auto' mode
  mode, params = _parse_mode(mode, params)
  # named args override params
  params = {**params, 'n': n, 'axis': axis}

  if mode == 'fftw':
    import pyfftw
    return pyfftw.interfaces.numpy_fft.irfft(a, **params)
  elif mode == 'np':
    return np.fft.irfft(a, **params)
  else:
    raise NotImplementedError('`irfft method is not defined for mode `%s`;' +
                              'use "np" or "fftw".')


def fhilbert(a, axis=None, mode='auto', ifft_params=None):
  """Compute the Hilbert transform of the provided frequency-space signal.

  This function assumes the input array is already in frequency space, i.e.,
  it is the output of a numpy-like FFT implementation. This avoids unnecessary
  repeated computation of the FFT/IFFT.

  Args:
    a (array): Signal, in frequency space, e.g., a = fft(signal).
    mode (str): Determines which FFT implementation will be used. Options are
      'fftw', 'np', and 'auto'. Using 'auto', will attempt to use a pyfftw
      implementation with some sensible parameters (if the module is
      available), and will use numpy's fftpack implementation otherwise.
    params (dict, None, optional): Dictionary of input arguments to provide to
      the call computing ifft. If `mode` is 'auto' and params dict is None,
      sensible values will be chosen. If `ifft_params` is not None, it will not
      be altered.

  Returns:
    array:
    **hilbert_a**: Hilbert transform of input array `a`, in the time domain.
  """
  if axis is None:
    axis = np.argmax(a.shape)
  N = a.shape[axis]
  if N <= 0:
    raise ValueError("N must be positive.")

  # perform the hilbert transform in the frequency domain
  # algorithm from scipy.signal.hilbert
  h = np.zeros(N)  # don't modify the input array
  # create hilbert multiplier
  if N % 2 == 0:
    h[0] = h[N // 2] = 1
    h[1:N // 2] = 2
  else:
    h[0] = 1
    h[1:(N + 1) // 2] = 2
  ah = a * h  # apply hilbert transform

  return ifft(ah, mode=mode, params=ifft_params)


def _parse_mode(mode, params):
  """Prepare mode and params arguments provided by user for use with
  utils.fft, utils.ifft, etc.

  Args:
    mode (str): Determines which FFT implementation will be used. Options are
      'fftw', 'np', and 'auto'. Using 'auto', will attempt to use a pyfftw
      implementation with some sensible parameters (if the module is
      available), and will use numpy's fftpack implementation otherwise.
    params (dict, None): Dictionary of input arguments to provide to the
      appropriate fft function. If `mode` is 'auto' and params dict is None,
      sensible values will be chosen. If `params` is not None, it will not be
      altered.

  Returns:
    tuple:
      **out_mode** (str): The mode determining the fft implementation to use; either
        'np' or 'fftw'.
      **out_params** (dict): A dictionary containing input arguments to the
        fft function.
  """
  mode == mode.lower()
  if mode == 'auto':
    try:
      import pyfftw
      mode = 'fftw'
      if params is None:
        params = {'planner_effort': 'FFTW_ESTIMATE'}  # FFTW_ESTIMATE seems fast
    except ImportError:
      mode = 'np'
      if params is None:
        params = {}
  else:
    if params is None:
      params = {}
  return mode, params
