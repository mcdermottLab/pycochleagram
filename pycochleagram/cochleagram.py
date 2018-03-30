
# TODO:
# + convert docstrings to np format
# + build and format docs
# + put docs on github
# + test padding (pad_factor)
# + sensible parameters for downsampling?
# + clean up old and deprecated methods
# + write readme
# + python compatibility issues
# + erb filters fails with certain arguments:
# `N: 680, sample_factor: 15, signal_length: 2433, sr: 32593, low_lim: 147, hi_lim: 16296, pad_factor: None`


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import sleep
import numpy as np
import scipy.signal

from pycochleagram import erbfilter as erb
from pycochleagram import subband as sb
import matplotlib.pyplot as plt

import pdb as ipdb


def cochleagram(signal, sr, n, low_lim, hi_lim, sample_factor,
        padding_size=None, downsample=None, nonlinearity=None,
        fft_mode='auto', ret_mode='envs', strict=True, **kwargs):
  """Generate the subband envelopes (i.e., the cochleagram)
  of the provided signal.

  This first creates a an ERB filterbank with the provided input arguments for
  the provided signal. This filterbank is then used to perform the subband
  decomposition to create the subband envelopes. The resulting envelopes can be
  optionally downsampled and then modified with a nonlinearity.

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
    padding_size (int, optional): If None (default), the signal will not be padded
      before filtering. Otherwise, the filters will be created assuming the
      waveform signal will be padded to length padding_size+signal_length.
    downsample (None, int, callable, optional): The `downsample` argument can
      be an integer representing the upsampling factor in polyphase resampling
      (with `sr` as the downsampling factor), a callable
      (to perform custom downsampling), or None to return the
      unmodified cochleagram; see `apply_envelope_downsample` for more
      information. If `ret_mode` is 'envs', this will be applied to the
      cochleagram before the nonlinearity, otherwise no downsampling will be
      performed. Providing a callable for custom downsampling is suggested.
    nonlinearity ({None, 'db', 'power', callable}, optional): The `nonlinearity`
      argument can be an predefined type, a callable
      (to apply a custom nonlinearity), or None to return the unmodified
      cochleagram; see `apply_envelope_nonlinearity` for more information.
      If `ret_mode` is 'envs', this will be applied to the cochleagram after
      downsampling, otherwise no nonlinearity will be applied. Providing a
      callable for applying a custom nonlinearity is suggested.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.
    ret_mode ({'envs', 'subband', 'analytic', 'all'}): Determines what will be
      returned. 'envs' (default) returns the subband envelopes; 'subband'
      returns just the subbands, 'analytic' returns the analytic signal provided
      by the Hilbert transform, 'all' returns all local variables created in this
      function.
    strict (bool, optional): If True (default), will include the extra
      highpass and lowpass filters required to make the filterbank invertible.
      If False, this will only perform calculations on the bandpass filters; note
      this decreases the number of frequency channels in the output by
       2 * `sample_factor`.
      function is used in a way that is unsupported by the MATLAB implemenation.
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

  # allow for batch generation without creating filters everytime
  batch_signal = sb.reshape_signal_batch(signal)  # (batch_dim, waveform_samples)

  # only make the filters once
  if kwargs.get('no_hp_lp_filts'):
    erb_kwargs = {'no_highpass': True, 'no_lowpass': True}
  else:
    erb_kwargs = {}
  # print(erb_kwargs)
  filts, hz_cutoffs, freqs = erb.make_erb_cos_filters_nx(batch_signal.shape[1],
      sr, n, low_lim, hi_lim, sample_factor, padding_size=padding_size,
      full_filter=True, strict=strict, **erb_kwargs)

  # utils.filtshow(freqs, filts, hz_cutoffs, use_log_x=True)

  freqs_to_plot = np.log10(freqs)

  # print(filts.shape)
  # plt.figure(figsize=(18,5))
  # # plt.plot(freqs_to_plot, filts[:,3:11], 'k')
  # plt.plot(freqs_to_plot, filts[:,5:13], 'k', linewidth=2)
  # plt.xlim([2, 3.5])
  # plt.ylim([0, None])
  # plt.title('%s @ %s' % (n, sample_factor))
  # wfn = '/om/user/raygon/projects/deepFerret/src/dflearn/COSYNE18_diagPlots/filters_%s_%s.pdf' % (n, sample_factor)
  # plt.savefig(wfn)
  # plt.show()
  # ipdb.set_trace()

  is_batch = batch_signal.shape[0] > 1
  for i in range(batch_signal.shape[0]):
    # if is_batch:
    #   print('generating cochleagram -> %s/%s' % (i+1, batch_signal.shape[0]))

    temp_signal_flat = sb.reshape_signal_canonical(batch_signal[i, ...])

    if ret_mode == 'envs' or ret_mode == 'all':
      temp_sb = sb.generate_subband_envelopes_fast(temp_signal_flat, filts,
          padding_size=padding_size, fft_mode=fft_mode, debug_ret_all=ret_all_sb)
    elif ret_mode == 'subband':
      temp_sb = sb.generate_subbands(temp_signal_flat, filts, padding_size=padding_size,
          fft_mode=fft_mode, debug_ret_all=ret_all_sb)
    elif ret_mode == 'analytic':
      temp_sb = sb.generate_subbands(temp_signal_flat, filts, padding_size=padding_size,
          fft_mode=fft_mode)
    else:
      raise NotImplementedError('`ret_mode` is not supported.')

    if ret_mode == 'envs':
      if downsample is None or callable(downsample):
        # downsample is None or callable
        temp_sb = apply_envelope_downsample(temp_sb, downsample)
      else:
        # interpret downsample as new sampling rate
        temp_sb = apply_envelope_downsample(temp_sb, 'poly', sr, downsample)
      temp_sb = apply_envelope_nonlinearity(temp_sb, nonlinearity)

    if i == 0:
      sb_out = np.zeros(([batch_signal.shape[0]] + list(temp_sb.shape)))
    sb_out[i] = temp_sb

  sb_out = sb_out.squeeze()
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
        sample_factor=2, padding_size=None, downsample=None, nonlinearity=None,
        fft_mode='auto', ret_mode='envs', strict=True, **kwargs):
  """Convenience function to generate the subband envelopes
  (i.e., the cochleagram) of the provided signal using sensible default
  parameters for a human cochleagram.

  This first creates a an ERB filterbank with the provided input arguments for
  the provided signal. This filterbank is then used to perform the subband
  decomposition to create the subband envelopes. The resulting envelopes can be
  optionally downsampled and then modified with a nonlinearity.

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
    padding_size (int, optional): If None (default), the signal will not be padded
      before filtering. Otherwise, the filters will be created assuming the
      waveform signal will be padded to length padding_size+signal_length.
    downsample (None, int, callable, optional): The `downsample` argument can
      be an integer representing the upsampling factor in polyphase resampling
      (with `sr` as the downsampling factor), a callable
      (to perform custom downsampling), or None to return the
      unmodified cochleagram; see `apply_envelope_downsample` for more
      information. If `ret_mode` is 'envs', this will be applied to the
      cochleagram before the nonlinearity, otherwise no downsampling will be
      performed. Providing a callable for custom downsampling is suggested.
    nonlinearity ({None, 'db', 'power', callable}, optional): The `nonlinearity`
      argument can be an predefined type, a callable
      (to apply a custom nonlinearity), or None to return the unmodified
      cochleagram; see `apply_envelope_nonlinearity` for more information.
      If `ret_mode` is 'envs', this will be applied to the cochleagram after
      downsampling, otherwise no nonlinearity will be applied. Providing a
      callable for applying a custom nonlinearity is suggested.
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
  if n is None:
    n = int(np.floor(erb.freq2erb(hi_lim) - erb.freq2erb(low_lim)) - 1)
  print("here")
  out = cochleagram(signal, sr, n, low_lim, hi_lim, sample_factor, padding_size,
      downsample, nonlinearity, fft_mode, ret_mode, strict, **kwargs)

  return out


def invert_cochleagram_with_filterbank(cochleagram, filters, sr, target_rms=100,
        downsample=None, nonlinearity=None, n_iter=20):
  """Generate a waveform from a cochleagram using a provided filterbank.

  Args:
    cochleagram (array): The subband envelopes (i.e., cochleagram) to invert.
    filters (array): The filterbank, in frequency space, used to generate the
      cochleagram. This should be the full filter-set output of
      erbFilter.make_erb_cos_filters_nx, or similar.
    sr (int): Sampling rate associated with the cochleagram.
    target_rms (scalar): Target root-mean-squared value of the output, related
      to SNR, TODO: this needs to be checked
    downsample (None, int, callable, optional): If downsampling was performed on
      `cochleagram`, this is the operation to invert that downsampling
      (i.e., upsample); this determines the length of the output signal.
      The `downsample` argument can be an integer representing the downsampling
      factor in polyphase resampling (with `sr` as the upsampling factor),
      a callable (to perform custom downsampling), or None to return the
      unmodified cochleagram; see `apply_envelope_downsample` for more
      information. Providing a callable for custom function for upsampling
      is suggested.
    nonlinearity ({None, 'db', 'power', callable}, optional): If a nonlinearity
      was applied to `cochleagram`, this is the operation to invert that
      nonlinearity.  The `nonlinearity` argument can be an predefined type,
      a callable (to apply a custom nonlinearity), or None to return the
      unmodified cochleagram; see `apply_envelope_nonlinearity` for more
      information. If this is a predefined type, the nonlinearity will be
      inverted according to `apply_envelope_nonlinearity`.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.
    n_iter (int, optional): Number of iterations to perform for the inversion.

  Returns:
    array:
    **inv_signal**: The waveform signal created by inverting the cochleagram.
  """
  # decompress envelopes
  linear_cochleagram = apply_envelope_nonlinearity(cochleagram, nonlinearity, invert=True)

  if downsample is None or callable(downsample):
    _wrapped_downsample = lambda coch, inv: apply_envelope_downsample(coch, downsample, invert=inv)  # downsample is None or callable
  else:
    # interpret downsample as new sampling rate
    _wrapped_downsample = lambda coch, inv: apply_envelope_downsample(coch, 'poly', sr, downsample, invert=inv)
  # apply the upsampling
  linear_cochleagram = _wrapped_downsample(cochleagram, True)

  coch_length = linear_cochleagram.shape[1]

  # cochleagram /= cochleagram.max()
  # print('ref coch: [%s, %s]' % (cochleagram.min(), cochleagram.max()))

  # generated signal starts from noise
  synth_size = coch_length
  synth_sound = np.random.random(synth_size)  # uniform noise
  # synth_sound = np.random.randn(synth_size)  # gaussian noise

  # print('synth sound [%s, %s]' % (synth_sound.min(), synth_sound.max()))

  # iteratively enforce envelopes on cochleagram of iter_noise
  for i in range(n_iter):
    # calculate error in decibels between original and synthesized cochleagrams
    # if i > 0:
    #   db_error = np.abs(cochleagram - np.abs(synth_analytic_subbands))
    # else:
    #   db_error = np.abs(cochleagram - np.zeros_like(cochleagram))

    # synth_sound = target_rms / utils.rms(synth_sound) * synth_sound

    # GET THE ERROR OF ENVS FROM DOWNSAMPLING
    synth_analytic_subbands = sb.generate_analytic_subbands(synth_sound, filters)
    synth_subband_mags = np.abs(synth_analytic_subbands)  # complex magnitude
    synth_subband_phases = synth_analytic_subbands / synth_subband_mags  # should be phases

    synth_subbands = synth_subband_phases * linear_cochleagram
    synth_subbands = np.real(synth_subbands)
    np.nan_to_num(synth_size)
    synth_sound = sb.collapse_subbands(synth_subbands, filters)

    synth_analytic_subbands = sb.generate_analytic_subbands(synth_sound, filters)
    synth_coch = np.abs(synth_analytic_subbands)

    # print('ref coch: [%s, %s], synth coch: [%s, %s]' % (cochleagram.min(), cochleagram.max(), synth_coch.min(), synth_coch.max()))

    # apply compression and downsample if necessary to compare reference coch to synth
    synth_coch = _wrapped_downsample(linear_cochleagram, False)
    synth_coch = apply_envelope_nonlinearity(synth_coch, nonlinearity, invert=False)

    # compute error using raw cochleagrams
    db_error = 10 * np.log10(np.sum(np.power(cochleagram - synth_coch, 2)) /
                np.sum(np.power(cochleagram, 2)))
    print('inverting iteration: %s, error (db): %s' % (i + 1, db_error))

  return synth_sound, synth_coch


def invert_cochleagram(cochleagram, sr, n, low_lim, hi_lim, sample_factor,
        padding_size=None, target_rms=100, downsample=None, nonlinearity=None, n_iter=50, strict=True):
  """Generate a waveform from a cochleagram using the provided arguments to
  construct a filterbank.

  Args:
    cochleagram (array): The subband envelopes (i.e., cochleagram) to invert.
    sr (int): Sampling rate associated with the cochleagram.
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
    padding_size (int, optional): If None (default), the signal will not be padded
      before filtering. Otherwise, the filters will be created assuming the
      waveform signal will be padded to length padding_size+signal_length.
    target_rms (scalar): Target root-mean-squared value of the output, related
      to SNR, TODO: this needs to be checked
    downsample (None, int, callable, optional): If downsampling was performed on
      `cochleagram`, this is the operation to invert that downsampling
      (i.e., upsample); this determines the length of the output signal.
      The `downsample` argument can be an integer representing the downsampling
      factor in polyphase resampling (with `sr` as the upsampling factor),
      a callable (to perform custom downsampling), or None to return the
      unmodified cochleagram; see `apply_envelope_downsample` for more
      information. Providing a callable for custom function for upsampling
      is suggested.
    nonlinearity ({None, 'db', 'power', callable}, optional): If a nonlinearity
      was applied to `cochleagram`, this is the operation to invert that
      nonlinearity.  The `nonlinearity` argument can be an predefined type,
      a callable (to apply a custom nonlinearity), or None to return the
      unmodified cochleagram; see `apply_envelope_nonlinearity` for more
      information. If this is a predefined type, the nonlinearity will be
      inverted according to `apply_envelope_nonlinearity`.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.
    n_iter (int, optional): Number of iterations to perform for the inversion.
    strict (bool, optional): If True (default), will throw an errors if this
      function is used in a way that is unsupported by the MATLAB implemenation.

  Returns:
    array:
    **inv_signal**: The waveform signal created by inverting the cochleagram.
    **inv_coch**: The inverted cochleagram.
  """
  # decompress envelopes
  cochleagram_ref = apply_envelope_nonlinearity(cochleagram, nonlinearity, invert=True)

  # upsample envelopes
  if downsample is None or callable(downsample):
    # downsample is None or callable
    cochleagram_ref = apply_envelope_downsample(cochleagram_ref, downsample, invert=True)
  else:
    # interpret downsample as new sampling rate
    cochleagram_ref = apply_envelope_downsample(cochleagram_ref, 'poly', sr, downsample, invert=True)
  signal_length = cochleagram_ref.shape[1]

# generate filterbank
  filts, hz_cutoffs, freqs = erb.make_erb_cos_filters_nx(signal_length,
      sr, n, low_lim, hi_lim, sample_factor, padding_size=padding_size,
      full_filter=True, strict=strict)

  # invert filterbank
  inv_signal, inv_coch = invert_cochleagram_with_filterbank(cochleagram_ref, filts, sr, target_rms=target_rms, n_iter=n_iter)

  return inv_signal, inv_coch


def apply_envelope_downsample(subband_envelopes, mode, audio_sr=None, env_sr=None, invert=False, strict=True):
  """Apply a downsampling operation to cochleagram subband envelopes.

  The `mode` argument can be a predefined downsampling type from
  {'poly', 'resample', 'decimate'}, a callable (to perform custom downsampling),
  or None to return the unmodified cochleagram. If `mode` is a predefined type,
  `audio_sr` and `env_sr` are required.

  Args:
    subband_envelopes (array): Cochleagram subbands to mode.
    mode ({'poly', 'resample', 'decimate', callable, None}): Determines the
      downsampling operation to apply to the cochleagram. 'decimate' will
      resample using scipy.signal.decimate with audio_sr/env_sr as the
      downsampling factor. 'resample' will downsample using
      scipy.signal.resample with np.ceil(subband_envelopes.shape[1]*(audio_sr/env_sr))
      as the number of samples. 'poly' will resample using scipy.signal.resample_poly
      with `env_sr` as the upsampling factor and `audio_sr` as the downsampling
      factor. If `mode` is a python callable (e.g., function), it will be
      applied to `subband_envelopes`. If this is None, no  downsampling is
      performed and the unmodified cochleagram is returned.
    audio_sr (int, optional): If using a predefined sampling `mode`, this
      represents the sampling rate of the original signal.
    env_sr (int, optional): If using a predefined sampling `mode`, this
      represents the sampling rate of the downsampled subband envelopes.
    invert (bool, optional):  If using a predefined sampling `mode`, this
      will invert (i.e., upsample) the subband envelopes using the values
      provided in `audio_sr` and `env_sr`.
    strict (bool, optional): If using a predefined sampling `mode`, this
      ensure the downsampling will result in an integer number of samples. This
      should mean the upsample(downsample(x)) will have the same number of
      samples as x.

  Returns:
    array:
    **downsampled_subband_envelopes**: The subband_envelopes after being
      downsampled with `mode`.
  """
  if mode is None:
    pass
  elif callable(mode):
    # apply the downsampling function
    subband_envelopes = mode(subband_envelopes)
  else:
    mode = mode.lower()
    if audio_sr is None:
      raise ValueError('`audio_sr` cannot be None. Provide sampling rate of original audio signal.')
    if env_sr is None:
      raise ValueError('`env_sr` cannot be None. Provide sampling rate of subband envelopes (cochleagram).')

    if mode == 'decimate':
      if invert:
        raise NotImplementedError()
      else:
        # was BadCoefficients error with Chebyshev type I filter [default]
        subband_envelopes = scipy.signal.decimate(subband_envelopes, audio_sr // env_sr, axis=1, ftype='fir') # this caused weird banding artifacts
    elif mode == 'resample':
      if invert:
        subband_envelopes = scipy.signal.resample(subband_envelopes, np.ceil(subband_envelopes.shape[1]*(audio_sr/env_sr)), axis=1)  # fourier method: this causes NANs that get converted to 0s
      else:
        subband_envelopes = scipy.signal.resample(subband_envelopes, np.ceil(subband_envelopes.shape[1]*(env_sr/audio_sr)), axis=1)  # fourier method: this causes NANs that get converted to 0s
    elif mode == 'poly':
      if strict:
        n_samples = subband_envelopes.shape[1] * (audio_sr / env_sr) if invert else subband_envelopes.shape[1] * (env_sr / audio_sr)
        if not np.isclose(n_samples, int(n_samples)):
          raise ValueError('Choose `env_sr` and `audio_sr` such that the number of samples after polyphase resampling is an integer'+
                           '\n(length: %s, env_sr: %s, audio_sr: %s !--> %s' % (subband_envelopes.shape[1], env_sr, audio_sr, n_samples))
      if invert:
        subband_envelopes = scipy.signal.resample_poly(subband_envelopes, audio_sr, env_sr, axis=1)  # this requires v0.18 of scipy
      else:
        subband_envelopes = scipy.signal.resample_poly(subband_envelopes, env_sr, audio_sr, axis=1)  # this requires v0.18 of scipy
    else:
      raise ValueError('Unsupported downsampling `mode`: %s' % mode)
  subband_envelopes[subband_envelopes < 0] = 0
  return subband_envelopes


def apply_envelope_nonlinearity(subband_envelopes, nonlinearity, invert=False):
  """Apply a nonlinearity to the cochleagram.

  The `nonlinearity` argument can be an predefined type, a callable
  (to apply a custom nonlinearity), or None to return the unmodified
  cochleagram.

  Args:
    subband_envelopes (array): Cochleagram to apply the nonlinearity to.
    nonlinearity ({'db', 'power'}, callable, None): Determines the nonlinearity
      operation to apply to the cochleagram. If this is a valid string, one
      of the predefined nonlinearities will be used. It can be: 'power' to
      perform np.power(subband_envelopes, 3.0 / 10.0) or 'db' to perform
      20 * np.log10(subband_envelopes / np.max(subband_envelopes)), with values
      clamped to be greater than -60. If `nonlinearity` is a python callable
      (e.g., function), it will be applied to `subband_envelopes`. If this is
      None, no nonlinearity is applied and the unmodified cochleagram is
      returned.
    invert (bool): For predefined nonlinearities 'db' and 'power', if False
      (default), the nonlinearity will be applied. If True, the nonlinearity
      will be inverted.

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
  elif nonlinearity == "db":
    if invert:
      subband_envelopes = np.power(10, subband_envelopes / 20)  # adapted from Anastasiya's code
    else:
      dtype_eps = np.finfo(subband_envelopes.dtype).eps
      subband_envelopes[subband_envelopes == 0] = dtype_eps
      subband_envelopes = 20 * np.log10(subband_envelopes / np.max(subband_envelopes))
      subband_envelopes[subband_envelopes < -60] = -60
  elif callable(nonlinearity):
    subband_envelopes = nonlinearity(subband_envelopes)
  else:
    raise ValueError('argument "nonlinearity" must be "power", "db", or a function.')
  return subband_envelopes
