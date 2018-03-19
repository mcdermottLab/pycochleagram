from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from scipy.io import wavfile
import warnings


def check_if_display_exists():
  """Check if a display is present on the machine. This can be used
  to conditionally import matplotlib, as importing it with an interactive
  backend on a machine without a display causes a core dump.

  Returns:
    (bool): Indicates if there is a display present on the machine.
  """
  havedisplay = 'DISPLAY' in os.environ
  if not havedisplay:
    exitval = os.system("python -c 'import matplotlib.pyplot as plt; plt.figure()'")
    havedisplay = (exitval == 0)
  return havedisplay


if check_if_display_exists():
  from matplotlib.pyplot import imshow, show, plot
else:
  import matplotlib
  matplotlib.use('Agg')
  from matplotlib.pyplot import imshow, show, plot
  warnings.warn('pycochleagram using non-interactive Agg matplotlib backend', RuntimeWarning, stacklevel=2)


##### Public Helper Methods #####
def compute_cochleagram_shape(signal_len, sr, n, sample_factor, env_sr=None):
  """Returns the shape of the cochleagram that will be created from
  by using the provided parameters.

  Args:
    signal_len (int): Length of signal waveform.
    sr (int): Waveform sampling rate.
    n (int): Number of filters requested in the filter bank.
    sample_factor (int): Degree of overcompleteness of the filter bank.
    env_sr (int, optional): Envelope sampling rate, if None (default),
      will equal the waveform sampling rate `sr`.

  Returns:
    tuple: Shape of the array containing the cochleagram.
  """
  env_sr = sr if env_sr is None else env_sr
  n_freqs = sample_factor * (n + 1) - 1 + 2 * sample_factor
  n_time = np.floor((env_sr / sr) * signal_len).astype(int)
  return (n_freqs, n_time)


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


def combine_signal_and_noise(signal, noise, snr):
  """Combine the signal and noise at the provided snr.

  Args:
    signal (array-like): Signal waveform data.
    noise (array-like): Noise waveform data.
    snr (number): SNR level in dB.

  Returns:
    **signal_and_noise**: Combined signal and noise waveform.
  """
  # normalize the signal
  signal = signal / rms(signal)
  sf = np.power(10, snr / 20)
  signal_rms = rms(signal)
  noise = noise * ((signal_rms / rms(noise)) / sf)
  signal_and_noise = signal + noise
  return signal_and_noise


def rms(a, strict=True):
  """Compute root mean squared of array.
  WARNING: THIS BREAKS WITH AXIS, only works on vector input.

  Args:
    a (array): Input array.

  Returns:
    array:
      **rms_a**: Root mean squared of array.
  """
  out = np.sqrt(np.mean(a * a))
  if strict and np.isnan(out):
    raise ValueError('rms calculation resulted in a nan: this will affect' +
                     'later computation. Ignore with `strict`=False')
  return out


##### Display and Playback Methods #####
def cochshow(cochleagram, interact=True, cmap='magma'):
  """Helper function to facilitate displaying cochleagrams.

  Args:
    cochleagram (array): Cochleagram to display with matplotlib.
    interact (bool, optional): Determines if interactive plot should be shown.
      If True (default), plot will be shown. If this is False, the figure will
      be created but not displayed.
    cmap (str, optional): A matplotlib cmap name to use for this plot.

  Returns:
    AxesImage:
    **image**: Whatever matplotlib.pyplot.plt returns.
  """
  f = imshow(cochleagram, aspect='auto', cmap=cmap, origin='lower', interpolation='nearest')
  if interact:
    show()
  return f


def filtshow(freqs, filts, hz_cutoffs=None, full_filter=True, use_log_x=False, interact=True):
  filts_to_plot = filts if full_filter is False else filts[:, :filts.shape[1]/2+1]  # positive filters
  # filts_to_plot = filts if full_filter is False else filts[:, filts.shape[1]/2-1:]
  freqs_to_plot = np.log10(freqs) if use_log_x else freqs

  print(filts_to_plot.shape)
  f = plot(freqs_to_plot, filts_to_plot.T)

  if hz_cutoffs is not None:
    hz_cutoffs_to_plot = np.log10(hz_cutoffs) if use_log_x else hz_cutoffs
    f = plot(hz_cutoffs_to_plot, np.zeros_like(hz_cutoffs)+filts_to_plot.max(), c='k', marker='o')

  if interact:
    show()
  return f


def get_channels(snd_array):
  """Returns the number of channels in the sound array.

  Args:
    snd_array (array): Array (of sound data).

  Returns:
    int:
    **n_channels**: The number of channels in the input array.
  """
  n_channels = 1
  if snd_array.ndim > 1:
    n_channels = snd_array.shape[1]
  return n_channels


def rescale_sound(snd_array, rescale):
  """Rescale the sound with the provided rescaling method (if supported).

  Args:
    snd_array (array): The array containing the sound data.
    rescale ({'standardize', 'normalize', None}): Determines type of
      rescaling to perform. 'standardize' will divide by the max value
      allowed by the numerical precision of the input. 'normalize' will
      rescale to the interval [-1, 1]. None will not perform rescaling (NOTE:
      be careful with this as this can be *very* loud if playedback!).

  Returns:
    array:
    **rescaled_snd**: The sound array after rescaling.
  """
  rescale = _parse_rescale_arg(rescale)
  if rescale == 'standardize':
    if issubclass(snd_array.dtype.type, np.integer):
      snd_array = snd_array / float(np.iinfo(snd_array.dtype).max)  # rescale so max value allowed by precision has value 1
    elif issubclass(snd_array.dtype.type, np.floating):
      snd_array = snd_array / float(np.finfo(snd_array.dtype).max)  # rescale so max value allowed by precision has value 1
    else:
      raise ValueError('rescale is undefined for input type: %s' % snd_array.dtype)
  elif rescale == 'normalize':
    snd_array = snd_array / float(snd_array.max())  # rescale to [-1, 1]
  # do nothing if rescale is None
  return snd_array


def wav_to_array(fn, rescale='standardize'):
  """ Reads wav file data into a numpy array.

    Args:
      fn (str): The file path to .wav file.
      rescale ({'standardize', 'normalize', None}): Determines type of
        rescaling to perform. 'standardize' will divide by the max value
        allowed by the numerical precision of the input. 'normalize' will
        rescale to the interval [-1, 1]. None will not perform rescaling (NOTE:
        be careful with this as this can be *very* loud if playedback!).

    Returns:
      tuple:
        **snd** (int): The sound in the .wav file as a numpy array.
        **samp_freq** (array): Sampling frequency of the input sound.
  """
  samp_freq, snd = wavfile.read(fn)
  snd = rescale_sound(snd, rescale)
  return snd, samp_freq


def play_array(snd_array, sr=44100, rescale='normalize', pyaudio_params={}, ignore_warning=False):
  """Play the provided sound array using pyaudio.

  Args:
    snd_array (array): The array containing the sound data.
    sr (number): Sampling sr for playback; defaults to 44,100 Hz.
    Will be overriden if `pyaudio_params` is provided.
    rescale ({'standardize', 'normalize', None}): Determines type of
      rescaling to perform. 'standardize' will divide by the max value
      allowed by the numerical precision of the input. 'normalize' will
      rescale to the interval [-1, 1]. None will not perform rescaling (NOTE:
      be careful with this as this can be *very* loud if playedback!).
    pyaudio_params (dict): A dictionary containing any input arguments to pass
      to the pyaudio.PyAudio.open method.
    ignore_warning (bool, optional): Determines if audio playback will occur.
      The playback volume can be very loud, so to use this method,
      `ignore_warning` must be True. If this is False, an error will be
      thrown warning the user about this issue.

  Returns:
    str:
      **sound_str**: The string representation (used by pyaudio) of the sound
        array.

  Raises:
    ValueError: If `ignore_warning` is False, an error is thrown to warn the
      user about the possible loud sounds associated with playback
  """
  import pyaudio
  if ignore_warning is not True:
    raise ValueError('WARNING: Playback is largely untested and can result in '+
        'VERY LOUD sounds. Use this function at your own risk. Dismiss this error '+
        'with `ignore_warning=True`.')

  out_snd_array = rescale_sound(snd_array, rescale)

  # _pyaudio_params = {'format': pyaudio.paFloat32,
  #                    'channels': 1,
  #                    'rate': sr,
  #                    'frames_per_buffer': 1024,
  #                    'output': True,
  #                    'output_device_index': 1}
  _pyaudio_params = {'format': pyaudio.paFloat32,
                   'channels': 1,
                   'rate': sr,
                   'frames_per_buffer': 1,  # I don't know what this does, but default of 1024 causes issues with TIMIT in py2.7
                   'output': True,
                   'output_device_index': 1}

  for k, v in pyaudio_params.items():
    _pyaudio_params[k] = v

  print('pyAudio Params:\n', _pyaudio_params)
  p = pyaudio.PyAudio()
  # stream = p.open(format=pyaudio.paFloat32,
  #                 channels=1,
  #                 rate=44100,
  #                 frames_per_buffer=1024,
  #                 output=True,
  #                 output_device_index=1)
  stream = p.open(**_pyaudio_params)
  data = out_snd_array.astype(np.float32).tostring()

  # stream = p.open(format=pyaudio.paInt16, channels=1, rate=samp_freq, output=True, frames_per_buffer=CHUNKSIZE)
  # data = snd.astype(snd.dtype).tostring()
  stream.write(data)
  return data


##### FFT-like Methods #####
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
  mode, params = _parse_fft_mode(mode, params)
  # named args override params
  d1 = {'n': n, 'axis': axis, 'norm': norm}
  params = dict(d1, **params)

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
  mode, params = _parse_fft_mode(mode, params)
  # named args override params
  d1 = {'n': n, 'axis': axis, 'norm': norm}
  params = dict(d1, **params)

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
  mode, params = _parse_fft_mode(mode, params)
  # named args override params
  d1 = {'n': n, 'axis': axis}
  params = dict(d1, **params)

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
  mode, params = _parse_fft_mode(mode, params)
  # named args override params
  # d1 = {'n': n, 'axis': axis, 'norm': norm}
  d1 = {'n': n, 'axis': axis}
  params = dict(d1, **params)

  if mode == 'fftw':
    import pyfftw
    return pyfftw.interfaces.numpy_fft.irfft(a, **params)
  elif mode == 'np':
    return np.fft.irfft(a, **params)
  else:
    raise NotImplementedError('`irfft method is not defined for mode `%s`;' +
                              'use "np" or "fftw".')


def hilbert(a, axis=None, mode='auto', fft_params=None):
  """Compute the Hilbert transform of time-domain signal.

  Provides access to FFTW-based implementation of the Hilbert transform.

  Args:
    a (array): Time-domain signal.
    mode (str): Determines which FFT implementation will be used. Options are
      'fftw', 'np', and 'auto'. Using 'auto', will attempt to use a pyfftw
      implementation with some sensible parameters (if the module is
      available), and will use numpy's fftpack implementation otherwise.
    fft_params (dict, None, optional): Dictionary of input arguments to provide to
      the call computing fft  and ifft. If `mode` is 'auto' and params dict is None,
      sensible values will be chosen. If `fft_params` is not None, it will not
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

  # convert to frequency space
  a = fft(a, mode=mode, params=fft_params)

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

  return ifft(ah, mode=mode, params=fft_params)


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
    iff_params (dict, None, optional): Dictionary of input arguments to provide to
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


##### Internal (Private) Helper Methods #####
def _parse_fft_mode(mode, params):
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


def _parse_rescale_arg(rescale):
  """Parse the rescaling argument to a standard form.

  Args:
    rescale ({'normalize', 'standardize', None}): Determines how rescaling
      will be performed.

  Returns:
    (str or None): A valid rescaling argument, for use with wav_to_array or
      similar.

  Raises:
    ValueError: Throws an error if rescale value is unrecognized.
  """
  if rescale is not None:
    rescale = rescale.lower()
  if rescale == 'normalize':
    out_rescale = 'normalize'
  elif rescale == 'standardize':
    out_rescale = 'standardize'
  elif rescale is None:
    out_rescale = None
  else:
    raise ValueError('Unrecognized rescale value: %s' % rescale)
  return out_rescale
