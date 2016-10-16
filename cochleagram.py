from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np
import scipy.signal

import erbfilter as erb
import subband as sb
import utils


def cochleagram(signal, sr, n, low_lim, hi_lim, sample_factor,
        pad_factor=None, downsample=None, nonlinearity=None,
        fft_mode='auto', ret_mode='envs', strict=True):
  """Generate the subband envelopes (i.e., the cochleagram)
  of the provided signal.

  This first creates a an ERB filterbank with the provided input arguments for
  the provided signal. This filterbank is then used to perform the subband
  decomposition to create the subband envelopes. The resulting envelopes can be
  optionally downsampled and then modified with a
  nonlinearity.

  To

  Args:
    signal (array): The sound signal (waveform) in the time domain. Should be
      flattened, i.e., the shape is (n_samples,).
    sr (int): Sampling rate associated with the signal waveform.
    n (int): Number of filters (subbands) to be generated with standard
      sampling (i.e., using a sampling factor of 1). Note, the actual number of
      filters in the generated filterbank depends on the sampling factor, and
      will also include lowpass and highpass filters that allow for
      perfect reconstruction of the input signal (the exact number of lowpass
      and highpass filters is determined by the sampling factor).
    low_lim (int): Lower limit of frequency range. Filters will not be defined
      below this limit.
    hi_lim (int): Upper limit of frequency range. Filters will not be defined
      above this limit.
    sample_factor (int): Positive integer that determines how densely ERB function
     will be sampled to create bandpass filters. 1 represents standard sampling;
     adjacent bandpass filters will overlap by 50%. 2 represents 2x overcomplete sampling;
     adjacent bandpass filters will overlap by 75%. 4 represents 4x overcomplete sampling;
     adjacent bandpass filters will overlap by 87.5%.
    pad_factor (int, optional): If None (default), the signal will not be padded
      before filtering. Otherwise, the filters will be created assuming the
      waveform signal will be padded to length pad_factor*signal_length.
    downsample (None, int, callable, optional): The `downsample` argument can
      be an downsampling factor, a callable (to perform custom downsampling),
      or None to return the unmodified cochleagram; see
      `apply_envelope_downsample` for more information. If `ret_mode` is
      'envs', this will be applied to the cochleagram before the nonlinearity.
      Providing a callable for custom downsampling is suggested.
    nonlinearity (None, int, callable, optional): The `nonlinearity` argument
      can be an predefined type, a callable (to apply a custom nonlinearity),
      or None to return the unmodified cochleagram; see
      `apply_envelope_nonlinearity` for more information. If `ret_mode` is
      'envs', this will be applied to the cochleagram after downsampling.
      Providing a callable for applying a custom nonlinearity is suggested.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.
    ret_mode ({'envs', 'subband', 'analytic', 'all'}): Determines what will be
      returned. 'envs' (default) returns the subband envelopes; 'subband'
      returns just the subbands, 'analytic' returns the analytic signal provided
      by the Hilber transform, 'all' returns all local variables created in this
      function.
    strict (bool, optional): If True (default), will throw an errors if this
      function is used in a way that is unsupported by the MATLAB implemenation.

  Returns:
    array:
    **out**: The output, depending on the value of `ret_mode`. If the `ret_mode`
      is 'envs' and a downsampling and/or nonlinearity
      operation was requested, the output will reflect these operations.
  """
  if strict:
    if not isinstance(sr, int):
      raise ValueError('`sr` must be an int; ignore with `strict`=False')
    # make sure low_lim and hi_lim are int
    if not isinstance(low_lim, int):
      raise ValueError('`low_lim` must be an int; ignore with `strict`=False')
    if not isinstance(hi_lim, int):
      raise ValueError('`hi_lim` must be an int; ignore with `strict`=False')

  ret_mode = ret_mode.lower()
  if ret_mode == 'all':
    ret_all_sb = True
  else:
    ret_all_sb = False

  # verify n is positive
  if n <= 0:
    raise ValueError('number of filters `n` must be positive; found: %s' % n)

  signal_flat = sb.reshape_signal_canonical(signal)

  filts, hz_cutoffs, freqs = erb.make_erb_cos_filters_nx(signal_flat.shape[0],
      sr, n, low_lim, hi_lim, sample_factor, pad_factor=pad_factor,
      full_filter=True, strict=strict)

  if ret_mode == 'envs' or ret_mode == 'all':
    sb_out = sb.generate_subband_envelopes_fast(signal, filts,
        pad_factor=pad_factor, downsample=downsample, nonlinearity=nonlinearity,
        fft_mode=fft_mode, debug_ret_all=ret_all_sb)
  elif ret_mode == 'subband':
    sb_out = sb.generate_subbands(signal, filts, pad_factor=pad_factor,
        fft_mode=fft_mode, debug_ret_all=ret_all_sb)
  elif ret_mode == 'analytic':
    sb_out = sb.generate_subbands(signal, filts, pad_factor=pad_factor,
        fft_mode=fft_mode)
  else:
    raise NotImplementedError('`ret_mode` is not supported.')

  if ret_mode == 'all':
    out_dict = {}
    # add all local variables to out_dict
    for k in dir():
      if k != 'out_dict':
        out_dict[k] = locals()[k]
    return out_dict
  else:
    return sb_out


def human_cochleagram(signal, sr, n=None, low_lim=50, hi_lim=20000,
        sample_factor=2, pad_factor=None, downsample=None, nonlinearity=None,
        fft_mode='auto', ret_mode='envs', strict=True):
  """Generate the subband envelopes (i.e., the cochleagram)
  of the provided signal using sensible default parameters for a human cochleagram.

  This first creates a an ERB filterbank with the provided input arguments for
  the provided signal. This filterbank is then used to perform the subband
  decomposition to create the subband envelopes. The resulting envelopes can be
  optionally downsampled and then modified with a
  nonlinearity.

  To

  Args:
    signal (array): The sound signal (waveform) in the time domain. Should be
      flattened, i.e., the shape is (n_samples,).
    sr (int): Sampling rate associated with the signal waveform.
    n (int): Number of filters (subbands) to be generated with standard
      sampling (i.e., using a sampling factor of 1). Note, the actual number of
      filters in the generated filterbank depends on the sampling factor, and
      will also include lowpass and highpass filters that allow for
      perfect reconstruction of the input signal (the exact number of lowpass
      and highpass filters is determined by the sampling factor).
    low_lim (int): Lower limit of frequency range. Filters will not be defined
      below this limit.
    hi_lim (int): Upper limit of frequency range. Filters will not be defined
      above this limit.
    sample_factor (int): Positive integer that determines how densely ERB function
     will be sampled to create bandpass filters. 1 represents standard sampling;
     adjacent bandpass filters will overlap by 50%. 2 represents 2x overcomplete sampling;
     adjacent bandpass filters will overlap by 75%. 4 represents 4x overcomplete sampling;
     adjacent bandpass filters will overlap by 87.5%.
    pad_factor (int, optional): If None (default), the signal will not be padded
      before filtering. Otherwise, the filters will be created assuming the
      waveform signal will be padded to length pad_factor*signal_length.
    downsample (None, int, callable, optional): The `downsample` argument can
      be an downsampling factor, a callable (to perform custom downsampling),
      or None to return the unmodified cochleagram; see
      `apply_envelope_downsample` for more information. If `ret_mode` is
      'envs', this will be applied to the cochleagram before the nonlinearity.
      Providing a callable for custom downsampling is suggested.
    nonlinearity (None, int, callable, optional): The `nonlinearity` argument
      can be an predefined type, a callable (to apply a custom nonlinearity),
      or None to return the unmodified cochleagram; see
      `apply_envelope_nonlinearity` for more information. If `ret_mode` is
      'envs', this will be applied to the cochleagram after downsampling.
      Providing a callable for applying a custom nonlinearity is suggested.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.
    ret_mode ({'envs', 'subband', 'analytic', 'all'}): Determines what will be
      returned. 'envs' (default) returns the subband envelopes; 'subband'
      returns just the subbands, 'analytic' returns the analytic signal provided
      by the Hilber transform, 'all' returns all local variables created in this
      function.
    strict (bool, optional): If True (default), will throw an errors if this
      function is used in a way that is unsupported by the MATLAB implemenation.

  Returns:
    array:
    **out**: The output, depending on the value of `ret_mode`. If the `ret_mode`
      is 'envs' and a downsampling and/or nonlinearity
      operation was requested, the output will reflect these operations.
  """
  signal_flat = sb.reshape_signal_canonical(signal)

  if n is None:
    n = int(np.floor(erb.freq2erb(hi_lim) - erb.freq2erb(low_lim)) - 1)

  out = cochleagram(signal_flat, sr, n, low_lim, hi_lim, sample_factor, pad_factor,
      downsample, nonlinearity, fft_mode, ret_mode, strict)

  return out


def batch_human_cochleagram(signal, sr, n=None, low_lim=50, hi_lim=20000,
        sample_factor=2, pad_factor=None, downsample=None, nonlinearity=None,
        fft_mode='auto', ret_mode='envs', strict=True):
  raise NotImplementedError()


def demo_human_cochleagram(signal=None):
  """Demo the cochleagram generation.

    signal (array, optional): If a time-domain signal is provided, its
      cochleagram will be generated with some sensible parameters. If this is
      None, a synthesized tone (harmonic stack of the first 40 harmonics) will
      be used.
  """
  if signal is None:
    dur = 50 / 1000
    sr = 20000
    env_sr = 6000
    pad_factor = 1
    q = sr // env_sr
    f0 = 100
    low_lim = 50
    hi_lim = 20000
    n = None


    # t = utils.matlab_arange(0, 200/1000, SR)
    t = np.arange(0, dur + 1 / sr, 1 / sr)
    signal = np.zeros_like(t)
    for i in range(1,40+1):
      signal += np.sin(2 * np.pi * f0 * i * t)

  downsample_fx = None
  # downsample_fx = lambda x: scipy.signal.resample_poly(x, 6000, sr, axis=1)
  # downsample_fx = lambda x: scipy.signal.decimate(x, 3, axis=1, ftype='fir') # this caused weird banding artifacts
  # downsample_fx = lambda x: scipy.signal.resample(x, np.ceil(x.shape[1]*(6000/sr)), axis=1)  # fourier method: this causes NANs that get converted to 0s

  nonlinearity_fx = None
  # nonlinearity_fx = 'log'
  # nonlinearity_fx = 'power'

  human_coch = human_cochleagram(signal, sr, strict=False)
  # human_coch = human_cochleagram(signal, sr, n=n, sample_factor=2,
  #     pad_factor=pad_factor, downsample=downsample_fx, nonlinearity=nonlinearity_fx,
  #     ret_mode='envs', strict=False)

  img = np.flipud(human_coch)
  print('sub env shape: ', human_coch.shape)
  utils.cochshow(img)

  return human_coch


def main():
  demo_human_cochleagram()


if __name__ == '__main__':
  main()
