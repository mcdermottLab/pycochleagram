from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
from time import sleep
import numpy as np
import scipy.signal

import erbfilter as erb
import subband as sb
import utils

import matplotlib.pyplot as plt


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
        pad_factor=pad_factor, fft_mode=fft_mode, debug_ret_all=ret_all_sb)
  elif ret_mode == 'subband':
    sb_out = sb.generate_subbands(signal, filts, pad_factor=pad_factor,
        fft_mode=fft_mode, debug_ret_all=ret_all_sb)
  elif ret_mode == 'analytic':
    sb_out = sb.generate_subbands(signal, filts, pad_factor=pad_factor,
        fft_mode=fft_mode)
  else:
    raise NotImplementedError('`ret_mode` is not supported.')

  if ret_mode == 'envs':
    sb_out = apply_envelope_downsample(sb_out, downsample)
    sb_out = apply_envelope_nonlinearity(sb_out, nonlinearity)

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


def invert_cochleagram_with_filterbank(cochleagram, filters, sr, env_sr=None, target_rms=100, downsample=None, nonlinearity=None, n_iter=5, test=None):
  coch_length = cochleagram.shape[1]
  # decompress envelopes
  print(nonlinearity)
  cochleagram = apply_envelope_nonlinearity(cochleagram, nonlinearity, invert=True)

  # # upsample
  # if env_sr is None:
  #   env_sr = sr
  # ds_factor = sr / env_sr
  # synth_size = ds_factor * coch_length
  # cochleagram = apply_envelope_downsample(cochleagram, downsample, invert=True)

  # dT = 1 / ds_factor
  # x = utils.matlab_arange(1, coch_length)
  # xI = utils.matlab_arange(dT, coch_length, dT)
  # upsampledEnv = interp1(x, xI, cochleagram)

  # generated signal starts from noise
  synth_size = coch_length
  print('inv coch sig size: ', synth_size)
  synth_sound = np.random.random(synth_size)  # uniform noise
  # synth_sound = np.random.randn(synth_size)  # gaussian noise

  target_subband_mags = target_rms  # don't know what this is

  # iteratively enforce envelopes on cochleagram of iter_noise
  for i in range(n_iter):
    if i % 100 == 0:
      if i > 0:
        plt.subplot(211)
        plt.title('Original Cochleagram')
        utils.cochshow(cochleagram, interact=False)
        plt.subplot(212)
        plt.title('Synth Cochleagram iter: %s' % i)
        utils.cochshow(np.abs(synth_analytic_subbands))
      test()
      sleep(1)
      utils.play_array(synth_sound, ignore_warning=True)
      sleep(1)

    print('inverting iteration: %s' % (i + 1))
    synth_sound = target_rms / utils.rms(synth_sound) * synth_sound

    # GET THE ERROR OF ENVS FROM DOWNSAMPLING
    synth_analytic_subbands = sb.generate_analytic_subbands(synth_sound, filters)
    synth_subband_mags = np.abs(synth_analytic_subbands)  # complex magnitude
    # synth_subband_phases = np.angle(synth_analytic_subbands)  # complex phases
    synth_subband_phases = synth_analytic_subbands / synth_subband_mags  # don't know what this is, from alex's code, converges faster

    synth_subbands = synth_subband_phases * cochleagram
    synth_subbands = np.real(synth_subbands)
    np.nan_to_num(synth_size)
    synth_sound = sb.collapse_subbands(synth_subbands, filters)


def invert_cochleagram(cochleagram, sr, n, low_lim, hi_lim, sample_factor,
        pad_factor=None, env_sr=None, downsample=None, nonlinearity=None, n_iter=1000, strict=True, test=None):
  print(locals())
  signal_length = cochleagram.shape[1]
  # generate filterbank
  filts, hz_cutoffs, freqs = erb.make_erb_cos_filters_nx(signal_length,
      sr, n, low_lim, hi_lim, sample_factor, pad_factor=pad_factor,
      full_filter=True, strict=strict)

  out_sig = invert_cochleagram_with_filterbank(cochleagram, filts, sr, n_iter=n_iter, nonlinearity=nonlinearity, test=test)

  return out_sig


def apply_envelope_downsample(subband_envelopes, downsample, env_sr=None, audio_sr=None, invert=False):
  """Apply a downsampling operation to cochleagram subband envelopes.

  The `downsample` argument can be an downsampling factor, a callable
  (to perform custom downsampling), or None to return the unmodified cochleagram.

  Args:
    subband_envelopes (array): Cochleagram subbands to downsample.
    downsample (int, callable, None): Determines the downsampling operation
      to apply to the cochleagram. If this is an int, assume that `downsample`
      represents the downsampling factor and apply scipy.signal.decimate to the
      cochleagram, over axis=1. If `downsample` is a python callable
      (e.g., function), it will be applied to `subband_envelopes`. If this is
      None, no  downsampling is performed and the unmodified cochleagram is
      returned.

  Returns:
    array:
    **downsampled_subband_envelopes**: The subband_envelopes after being
      downsampled with `downsample`.
  """
  if downsample is None:
    pass
  elif callable(downsample):
    # apply the downsampling function
    subband_envelopes = downsample(subband_envelopes)
  else:
    downsample = downsample.lower()
    if audio_sr is None:
      raise ValueError('`audio_sr` cannot be None. Provide sampling rate of original audio signal.')
    if env_sr is None:
      raise ValueError('`env_sr` cannot be None. Provide sampling rate of subband envelopes (cochleagram).')

    if downsample == 'decimate':
      if invert:
        raise NotImplementedError()
      else:
        # assume that downsample is the downsampling factor
        # was BadCoefficients error with Chebyshev type I filter [default]
        subband_envelopes = scipy.signal.decimate(subband_envelopes, audio_sr // env_sr, axis=1, ftype='fir') # this caused weird banding artifacts
    if downsample == 'resample':
      if invert:
        subband_envelopes = scipy.signal.resample(subband_envelopes, np.ceil(subband_envelopes.shape[1]*(audio_sr/env_sr)), axis=1)  # fourier method: this causes NANs that get converted to 0s
      else:
        subband_envelopes = scipy.signal.resample(subband_envelopes, np.ceil(subband_envelopes.shape[1]*(env_sr/audio_sr)), axis=1)  # fourier method: this causes NANs that get converted to 0s
    if downsample == 'poly':
      if invert:
        subband_envelopes = scipy.signal.resample_poly(subband_envelopes, audio_sr, env_sr, axis=1)  # this requires v0.18 of scipy
      else:
        subband_envelopes = scipy.signal.resample_poly(subband_envelopes, env_sr, audio_sr, axis=1)  # this requires v0.18 of scipy
  subband_envelopes[subband_envelopes < 0] = 0
  return subband_envelopes


def apply_envelope_nonlinearity(subband_envelopes, nonlinearity, invert=False):
  """Apply a nonlinearity to the cochleagram.

  The `nonlinearity` argument can be an predefined type, a callable
  (to apply a custom nonlinearity), or None to return the unmodified
  cochleagram.

  Args:
    subband_envelopes (array): Cochleagram to apply the nonlinearity to.
    nonlinearity (str, callable, None): Determines the nonlinearity operation
      to apply to the cochleagram. If this is a valid string, one of the
      predefined nonlinearities will be used. It can be: 'power' to perform
      np.power(subband_envelopes, 3.0 / 10.0) or 'log' to perform
      20 * np.log10(subband_envelopes / np.max(subband_envelopes)).
      If `nonlinearity` is a python callable (e.g., function), it will be
      applied to `subband_envelopes`. If this is None, no nonlinearity is
      applied and the unmodified cochleagram is returned.

  Returns:
    array:
    **nonlinear_subband_envelopes**: The subband_envelopes with the specified
      nonlinearity applied.

  Raises:
      ValueError: Error if the provided `nonlinearity` isn't a recognized
      option.
  """
  # apply nonlinearity
  if nonlinearity is None:
    pass
  elif nonlinearity == "power":
    if invert:
      subband_envelopes = np.power(subband_envelopes, 10.0 / 3.0)  # from Alex's code
    else:
      subband_envelopes = np.power(subband_envelopes, 3.0 / 10.0)  # from Alex's code
  elif nonlinearity == "log":
    if invert:
      # subband_envelopes = np.log(subband_envelopes)  # adapted from Alex's code
      subband_envelopes = np.power(10, subband_envelopes / 20)  # adapted from Anastasiya's code
    else:
      dtype_eps = np.finfo(subband_envelopes.dtype).eps
      # subband_envelopes[subband_envelopes <= 0] += dtype_eps
      subband_envelopes[subband_envelopes == 0] = dtype_eps
      # subband_envelopes = np.log(subband_envelopes)  # adapted from Alex's code
      subband_envelopes = 20 * np.log10(subband_envelopes / np.max(subband_envelopes))  # adapted from Anastasiya's code
      # a = subband_envelopes / np.max(subband_envelopes)
      # a[a <= 0] = dtype_eps
      # subband_envelopes = 20 * np.log10(a)  # adapted from Anastasiya's code
  elif callable(nonlinearity):
    subband_envelopes = nonlinearity(subband_envelopes)
  else:
    raise ValueError('argument "nonlinearity" must be "power", "log", or a function.')
  return subband_envelopes


def demo_human_cochleagram(signal=None, sr=None, downsample=None, nonlinearity=None, interact=True):
  """Demo the cochleagram generation.

    signal (array, optional): If a time-domain signal is provided, its
      cochleagram will be generated with some sensible parameters. If this is
      None, a synthesized tone (harmonic stack of the first 40 harmonics) will
      be used.
    downsample({'poly', 'resample', 'decimate', None}): Determines downsampling
      method to apply.
    nonlinearity({'log', 'power', None}): Determines nonlinearity method to
      apply.
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
    # print(signal.shape)

  if downsample is None:
    downsample_fx = None
  elif downsample == 'poly':
    downsample_fx = lambda x: scipy.signal.resample_poly(x, env_sr, sr, axis=1)
  elif downsample == 'resample':
    downsample_fx = lambda x: scipy.signal.resample(x, np.ceil(x.shape[1]*(env_sr/sr)), axis=1)  # fourier method: this causes NANs that get converted to 0s
  elif downsample == 'decimate':
    downsample_fx = lambda x: scipy.signal.decimate(x, q, axis=1, ftype='fir') # this caused weird banding artifacts
  else:
    raise NotImplementedError()

  if nonlinearity is None:
    nonlinearity_fx = None
  elif nonlinearity == 'log':
    nonlinearity_fx = 'log'
  elif nonlinearity == 'power':
    nonlinearity_fx = 'power'
  else:
    raise NotImplementedError()

  # human_coch = human_cochleagram(signal, sr, strict=False)
  human_coch = human_cochleagram(signal, sr, n=n, sample_factor=2,
      pad_factor=pad_factor, downsample=downsample_fx, nonlinearity=nonlinearity_fx,
      ret_mode='envs', strict=False)

  img = np.flipud(human_coch)
  if interact:
    print('sub env shape: ', human_coch.shape)
    utils.cochshow(img)

  return img, {'signal': signal, 'sr': sr}


def demo_invert_cochleagram(signal=None, sr=None):
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
    if n is None:
      n = int(np.floor(erb.freq2erb(hi_lim) - erb.freq2erb(low_lim)) - 1)

    sample_factor = 2

    # t = utils.matlab_arange(0, 200/1000, SR)
    t = np.arange(0, dur + 1 / sr, 1 / sr)
    signal = np.zeros_like(t)
    for i in range(1,40+1):
      signal += np.sin(2 * np.pi * f0 * i * t)

  downsample_fx = None
  # downsample_fx = 'poly'
  env_sr = 6000

  nonlinearity_fx = None
  # nonlinearity_fx = 'log'
  # nonlinearity_fx = 'power'

  coch = human_cochleagram(signal, sr, downsample=downsample_fx, nonlinearity=nonlinearity_fx, strict=False)
  # coch = np.flipud(coch)
  downsample_fx = 'poly'
  plt.subplot(311)
  utils.cochshow(coch, interact=False)
  print(coch.shape)
  plt.subplot(312)
  coch = apply_envelope_downsample(coch, downsample_fx, env_sr, sr, invert=False)
  # # coch = coch ** (3/10)
  # max_coch = coch.max()
  # coch = 20 * np.log10(coch / coch.max())
  utils.cochshow(coch, interact=False)
  plt.subplot(313)
  # inv_coch = apply_envelope_nonlinearity(coch, nonlinearity_fx, invert=True)
  inv_coch = apply_envelope_downsample(coch, downsample_fx, env_sr, sr, invert=True)
  # # inv_coch = coch ** (10/3)
  # inv_coch = np.power(10, coch / 20)
  print(inv_coch.shape)
  utils.cochshow(inv_coch)

  test = lambda: utils.play_array(signal, ignore_warning=True)
  invert_cochleagram(coch, sr, n, low_lim, hi_lim, sample_factor, pad_factor=None, nonlinearity=nonlinearity_fx, strict=False, test=test)


def main():
  demo_invert_cochleagram()


if __name__ == '__main__':
  main()
