from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

import erbfilter as erb
import utils


def reshape_signal_canonical(signal):
  """Convert the signal into a canonical shape for use with cochleagram.py
  functions.

  This first verifies that the signal contains only one data channel, which can
  be in a row, a column, or a flat array. Then it flattens the signal array.

  Args:
    signal (array): The sound signal (waveform) in the time domain. Should be
      either a flattened array with shape (n_samples,), a row vector with shape
      (1, n_samples), or a column vector with shape (n_samples, 1).

  Returns:
    array:
    **out_signal**: If the input `signal` has a valid shape, returns a
      flattened version of the signal.

  Raises:
    ValueError: Raises an error of the input `signal` has invalid shape.
  """
  if signal.ndim == 1:  # signal is a flattened array
    out_signal = signal
  elif signal.ndim == 2:  # signal is a row or column vector
    if signal.shape[0] == 1:
      out_signal = signal.flatten()
    elif signal.shape[1] == 1:
      out_signal = signal.flatten()
    else:
      raise ValueError('signal must be a row or column vector; found shape: %s' % signal.shape)
  else:
    raise ValueError('signal must be a row or column vector; found shape: %s' % signal.shape)
  return out_signal


def generate_subband_envelopes_fast(signal, filters, pad_factor=None,
      downsample=None, nonlinearity=None, fft_mode='auto', debug_ret_all=False):
  """Generate the subband envelopes (i.e., the cochleagram) of the signal by
  applying the provided filters.

  This method returns *only* the envelopes of the subband decomposition.
  The signal can be optionally zero-padded before the decomposition. The
  resulting envelopes can be optionally downsampled and then modified with a
  nonlinearity.

  This function expedites the calculation of the subbands envelopes by:
    1) using the rfft rather than standard fft to compute the dft for
       real-valued signals
    2) hand-computing the Hilbert transform, to avoid unnecessary calls
       to fft/ifft.

  See utils.rfft, utils.irfft, and utils.fhilbert for more details on the
  methods used for speed-up.

  Args:
    signal (array): The sound signal (waveform) in the time domain. Should be
      flattened, i.e., the shape is (n_samples,).
    filters (array): The filterbank, in frequency space, used to generate the
      cochleagram. This should be the full filter-set output of
      erbFilter.make_erb_cos_filters_nx, or similar.
    pad_factor (int, optional): Factor that determines if the signal will be
      zero-padded before generating the subbands. If this is None,
      or less than 1, no zero-padding will be used. Otherwise, zeros are added
      to the end of the input signal until is it of length
      `pad_factor * length(signal)`. This padded region will be removed after
      performing the subband decomposition.
    downsample (None, int, callable, optional): The `downsample` argument can
      be an downsampling factor, a callable (to perform custom downsampling),
      or None to return the unmodified cochleagram; see
      `apply_envelope_downsample` for more information. This will be applied
      to the cochleagram before the nonlinearity. Providing a callable for
      custom downsampling is suggested.
    nonlinearity (None, int, callable, optional): The `nonlinearity` argument
      can be an predefined type, a callable (to apply a custom nonlinearity),
      or None to return the unmodified cochleagram; see
      `apply_envelope_nonlinearity` for more information. This will be applied
      to the cochleagram after downsampling. Providing a callable for applying
      a custom nonlinearity is suggested.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.

  Returns:
    array:
    **subband_envelopes**: The subband envelopes (i.e., cochleagram) resulting from
      the subband decomposition. If a downsampling and/or nonlinearity
      operation was requested, the output will reflect these operations.
      This should have the same shape as `filters`.
  """
  # convert the signal to a canonical representation
  signal_flat = reshape_signal_canonical(signal)

  if pad_factor is not None and pad_factor > 1:
    signal_flat, padding = pad_signal(signal_flat, pad_factor)

  if np.isrealobj(signal_flat):  # attempt to speed up computation with rfft
    fft_sample = utils.rfft(signal_flat, mode=fft_mode)
    nr = fft_sample.shape[0]
    # prep for hilbert transform by extending to negative freqs
    subbands = np.zeros(filters.shape, dtype=complex)
    subbands[:, :nr] = _real_freq_filter(fft_sample, filters)
  else:
    fft_sample = utils.fft(signal_flat, mode=fft_mode)
    subbands = filters * fft_sample

  analytic_subbands = utils.fhilbert(subbands, mode=fft_mode)
  subband_envelopes = np.abs(analytic_subbands)

  if pad_factor is not None and pad_factor > 1:
    analytic_subbands = analytic_subbands[:, :signal_flat.shape[0] - padding]  # i dont know if this is correct
    subband_envelopes = subband_envelopes[:, :signal_flat.shape[0] - padding]  # i dont know if this is correct

  subband_envelopes = apply_envelope_downsample(subband_envelopes, downsample)
  subband_envelopes = apply_envelope_nonlinearity(subband_envelopes, nonlinearity)

  if debug_ret_all is True:
    out_dict = {}
    # add all local variables to out_dict
    for k in dir():
      if k != 'out_dict':
        out_dict[k] = locals()[k]
    return out_dict
  else:
    return subband_envelopes


def generate_subband_envelopes_alex_fast(signal, filters, pad_factor=None,
      downsample=None, nonlinearity=None, debug_ret_all=False):
  """Generate the subband envelopes (i.e., the cochleagram) of the signal by
  applying the provided filters.

  This method returns *only* the envelopes of the subband decomposition.
  The signal can be optionally zero-padded before the decomposition. The
  resulting envelopes can be optionally downsampled and then modified with a
  nonlinearity.

  This function expedites the calculation of the subbands envelopes by:
    1) using the rfft rather than standard fft to compute the dft.
    2) hand-computing the Hilbert transform, to avoid repeated and unnecessary
       calls to fft/ifft.

  The Fourier-based analytic signal algorithm can be found here:
  Marple.  Computing the Discrete-Time Analytic Signal via FFT.  IEEE Trans Sig Proc 1999.
  http://classes.engr.oregonstate.edu/eecs/winter2009/ece464/AnalyticSignal_Sept1999_SPTrans.pdf

  Credit to Alex Kell for this method.

  Args:
    signal (array): The sound signal (waveform) in the time domain. Should be
      flattened, i.e., the shape is (n_samples,).
    filters (array): The filterbank, in frequency space, used to generate the
      cochleagram. This should be the full filter-set output of
      erbFilter.make_erb_cos_filters_nx, or similar.
    pad_factor (int, optional): Factor that determines if the signal will be
      zero-padded before generating the subbands. If this is None,
      or less than 1, no zero-padding will be used. Otherwise, zeros are added
      to the end of the input signal until is it of length
      `pad_factor * length(signal)`. This padded region will be removed after
      performing the subband decomposition.
    downsample (None, int, callable, optional): The `downsample` argument can
      be an downsampling factor, a callable (to perform custom downsampling),
      or None to return the unmodified cochleagram; see
      `apply_envelope_downsample` for more information. This will be applied
      to the cochleagram before the nonlinearity. Providing a callable for
      custom downsampling is suggested.
    nonlinearity (None, int, callable, optional): The `nonlinearity` argument
      can be an predefined type, a callable (to apply a custom nonlinearity),
      or None to return the unmodified cochleagram; see
      `apply_envelope_nonlinearity` for more information. This will be applied
      to the cochleagram after downsampling. Providing a callable for applying
      a custom nonlinearity is suggested.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.

  Returns:
    array:
    **subband_envelopes**: The subband envelopes (i.e., cochleagram) resulting from
      the subband decomposition. If a downsampling and/or nonlinearity
      operation was requested, the output will reflect these operations.
      This should have the same shape as `filters`.
  """
  warnings.warn('Function is deprecated; use generate_subband_envelopes_fast instead', DeprecationWarning)

  # convert the signal to a canonical representation
  signal_flat = reshape_signal_canonical(signal)

  if pad_factor is not None and pad_factor >= 1:
    signal_flat, padding = pad_signal(signal_flat, pad_factor)

  n = signal_flat.shape[0]
  assert np.mod(n, 2) == 0  # likely overly stringent but just for safety here

  nr = int(np.floor(n / 2)) + 1
  n_filts = filters.shape[0]

  # note: the real fft, signal needs to be flat or have axis specified
  Fr = np.fft.rfft(signal_flat)  # len: nr = n/2+1 (e.g., 8001 for 1 sec and 16K sr)

  # pdb.set_trace()

  # compute the subbands
  # note that "real" here doesn't indicate that the variable consists of real
  # values; it doesn't -- the values are complex
  # instead, "real" indicates that it's half of the full DFT, just the positive
  # frequencies because the negatives are redundant for real-valued sigs
  subbands_fourier_real = filters[:, :nr] * Fr

  # manually compute the hilbert transform here
  # h is a vector which:
  # -- doubles the amplitude of the real positive frequencies
  # -- leaves the DC and the postive and negative nyquist unchanged
  # -- zeros out the negative frequencies (which are already zeroed here
  h = np.zeros(nr)
  h[0], h[1:n/2], h[n/2] = 1.0, 2.0, 1.0  # remember for middle, noninclusive upper bound

  analytic_subbands_fourier_real = subbands_fourier_real * h

  ## fill this in so can make a simple ifft call
  ## can't make an irfft call because we DON'T want to assume that
  ## the negative frequencies are conjugate
  ## the time-domain analytic signal is complex
  ## (the negative frequencies here are in fact they are zero)
  analytic_subbands_fourier = np.zeros((n_filts, n), dtype=complex)
  analytic_subbands_fourier[:, :nr] = analytic_subbands_fourier_real

  ## below is in the time domain
  analytic_subbands = np.fft.ifft(analytic_subbands_fourier, axis=1)
  subband_envelopes = np.abs(analytic_subbands)

  if pad_factor is not None and pad_factor >= 1:
    analytic_subbands = analytic_subbands[:, :signal_flat.shape[0] - padding]  # i dont know if this is correct
    subband_envelopes = subband_envelopes[:, :signal_flat.shape[0] - padding]  # i dont know if this is correct

  subband_envelopes = apply_envelope_downsample(subband_envelopes, downsample)
  subband_envelopes = apply_envelope_nonlinearity(subband_envelopes, nonlinearity)

  if debug_ret_all is True:
    out_dict = {}
    # add all local variables to out_dict
    for k in dir():
      if k != 'out_dict':
        out_dict[k] = locals()[k]
    return out_dict
  else:
    return subband_envelopes


def generate_subbands(signal, filters, pad_factor=None, fft_mode='auto', debug_ret_all=False):
  """Generate the subband decomposition of the signal by applying the provided
  filters.

  The input filters are applied to the signal to perform subband decomposition.
  The signal can be optionally zero-padded before the decomposition.

  Args:
    signal (array): The sound signal (waveform) in the time domain.
    filters (array): The filterbank, in frequency space, used to generate the
      cochleagram. This should be the full filter-set output of
      erbFilter.make_erb_cos_filters_nx, or similar.
    pad_factor (int, optional): Factor that determines if the signal will be
      zero-padded before generating the subbands. If this is None,
      or less than 1, no zero-padding will be used. Otherwise, zeros are added
      to the end of the input signal until is it of length
      `pad_factor * length(signal)`. This padded region will be removed after
      performing the subband decomposition.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.

  Returns:
    array:
    **subbands**: The subbands resulting from the subband decomposition. This
      should have the same shape as `filters`.
  """
  # note: numpy defaults to row vecs
  # if pad_factor is not None and pad_factor >= 1:
  #   padding = signal.shape[0] * pad_factor - signal.shape[0]
  #   print('padding ', padding)
  #   signal = np.concatenate((signal, np.zeros(padding)))

  # convert the signal to a canonical representation
  signal_flat = reshape_signal_canonical(signal)

  if pad_factor is not None and pad_factor > 1:
    signal_flat, padding = pad_signal(signal_flat, pad_factor)

  is_signal_even = signal_flat.shape[0] % 2 == 0
  if np.isrealobj(signal_flat) and is_signal_even:  # attempt to speed up computation with rfft
    if signal_flat.shape[0] % 2 == 0:
      fft_sample = utils.rfft(signal_flat, mode=fft_mode)
      subbands = _real_freq_filter(fft_sample, filters)
      subbands = utils.irfft(subbands, mode=fft_mode)  # operates row-wise
    else:
      warnings.warn('Consider using even-length signal for a rfft speedup', RuntimeWarning, stacklevel=2)
      fft_sample = utils.fft(signal_flat, mode=fft_mode)
      subbands = filters * fft_sample
      subbands = np.real(utils.ifft(subbands, mode=fft_mode))  # operates row-wise
  else:
    fft_sample = utils.fft(signal_flat, mode=fft_mode)
    subbands = filters * fft_sample
    subbands = np.real(utils.ifft(subbands, mode=fft_mode))  # operates row-wise

  if pad_factor is not None and pad_factor > 1:
    subbands = subbands[:, :signal_flat.shape[0] - padding]  # i dont know if this is correct

  if debug_ret_all is True:
    out_dict = {}
    # add all local variables to out_dict
    for k in dir():
      if k != 'out_dict':
        out_dict[k] = locals()[k]
    return out_dict
  else:
    return subbands


def generate_analytic_subbands(signal, filters, pad_factor=None, fft_mode='auto'):
  """Generate the analytic subbands (i.e., hilbert transform) of the signal by
    applying the provided filters.

    The input filters are applied to the signal to perform subband decomposition.
    The signal can be optionally zero-padded before the decomposition. For full
    cochleagram generation, see generate_subband_envelopes.

  Args:
    signal (array): The sound signal (waveform) in the time domain.
    filters (array): The filterbank, in frequency space, used to generate the
      cochleagram. This should be the full filter-set output of
      erbFilter.make_erb_cos_filters_nx, or similar.
    pad_factor (int, optional): Factor that determines if the signal will be zero-padded
      before generating the subbands. If this is None, or less than 1, no
      zero-padding will be used. Otherwise, zeros are added to the end of the
      input signal until is it of length `pad_factor * length(signal)`. This
      padded region will be removed after performing the subband
      decomposition.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.
      TODO: fix zero-padding

  Returns:
    array:
    **analytic_subbands**: The analytic subbands (i.e., hilbert transform) resulting
      of the subband decomposition. This should have the same shape as
      `filters`.
  """
  subbands = generate_subbands(signal, filters, pad_factor=pad_factor, fft_mode=fft_mode)
  # subbands = subbands[:, :signal.shape[0]/2]
  return scipy.signal.hilbert(subbands)


def generate_subband_envelopes(signal, filters, pad_factor=None, downsample=None, nonlinearity=None, debug_ret_all=False):
  """Generate the subband envelopes (i.e., the cochleagram) of the signal by
    applying the provided filters.

  The input filters are applied to the signal to perform subband decomposition.
  The signal can be optionally zero-padded before the decomposition. The
  resulting cochleagram can be optionally downsampled and then modified with a
  nonlinearity.

  Args:
    signal (array): The sound signal (waveform) in the time domain.
    filters (array): The filterbank, in frequency space, used to generate the
      cochleagram. This should be the full filter-set output of
      erbFilter.make_erb_cos_filters_nx, or similar.
    pad_factor (int, optional): Factor that determines if the signal will be zero-padded
      before generating the subbands. If this is None, or less than 1, no
      zero-padding will be used. Otherwise, zeros are added to the end of the
      input signal until is it of length `pad_factor * length(signal)`. This
      padded region will be removed after performing the subband
      decomposition.
    downsample (None, int, callable, optional): The `downsample` argument can
      be an downsampling factor, a callable (to perform custom downsampling),
      or None to return the unmodified cochleagram; see
      apply_envelope_downsample for more information. This will be applied
      to the cochleagram before the nonlinearity. Providing a callable for
      custom downsampling is suggested.
    nonlinearity (None, int, callable, optional): The `nonlinearity` argument
      can be an predefined type, a callable (to apply a custom nonlinearity),
      or None to return the unmodified cochleagram; see
      apply_envelope_nonlinearity for more information. This will be applied
      to the cochleagram after downsampling. Providing a callable for applying
      a custom nonlinearity is suggested.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.

  Returns:
    array:
    **subband_envelopes**: The subband envelopes (i.e., cochleagram) resulting from
      the subband decomposition. If a downsampling and/or nonlinearity
      operation was requested, the output will reflect these operations.
      This should have the same shape as `filters`.
  """
  analytic_subbands = generate_analytic_subbands(signal, filters, pad_factor=pad_factor)
  subband_envelopes = np.abs(analytic_subbands)

  subband_envelopes = apply_envelope_downsample(subband_envelopes, downsample)
  subband_envelopes = apply_envelope_nonlinearity(subband_envelopes, nonlinearity)

  if debug_ret_all is True:
    out_dict = {}
    # add all local variables to out_dict
    for k in dir():
      if k != 'out_dict':
        out_dict[k] = locals()[k]
    return out_dict
  else:
    return subband_envelopes


def pad_signal(signal, pad_factor, axis=0):
  """Pad the signal by appending zeros to the end. The padded signal has
  length `pad_factor * length(signal)`.

  Args:
    signal (array): The signal to be zero-padded.
    pad_factor (int): Factor that determines the size of the padded signal.
      The padded signal has length `pad_factor * length(signal)`.
    axis (int): Specifies the axis to pad; defaults to 0.

  Returns:
    tuple:
      **pad_signal** (*array*): The zero-padded signal.
      **padding_size** (*int*): The length of the zero-padding added to the array.
  """
  if pad_factor is not None and pad_factor >= 1:
    padding_size = signal.shape[axis] * pad_factor - signal.shape[axis]
    pad_shape = list(signal.shape)
    pad_shape[axis] = padding_size
    pad_signal = np.concatenate((signal, np.zeros(pad_shape)))
  else:
    padding_size = 0
    pad_signal = signal
  return (pad_signal, padding_size)


def apply_envelope_downsample(subband_envelopes, downsample):
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
    # assume that downsample is the downsampling factor
    # was BadCoefficients error with Chebyshev type I filter [default]
    #   resample uses a fourier method and is needlessly long...
    subband_envelopes = scipy.signal.decimate(subband_envelopes, downsample, axis=1, ftype='fir') # this caused weird banding artifacts
    # subband_envelopes = scipy.signal.resample(subband_envelopes, np.ceil(subband_envelopes.shape[1]*(6000/SR)), axis=1)  # fourier method: this causes NANs that get converted to 0s
    # subband_envelopes = scipy.signal.resample_poly(subband_envelopes, 6000, SR, axis=1)  # this requires v0.18 of scipy
  subband_envelopes[subband_envelopes < 0] = 0
  return subband_envelopes


def apply_envelope_nonlinearity(subband_envelopes, nonlinearity):
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
    subband_envelopes = np.power(subband_envelopes, 3.0 / 10.0)  # from Alex's code
  elif nonlinearity == "log":
    print(subband_envelopes.dtype)
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


def _real_freq_filter(rfft_signal, filters):
  """Helper function to apply a full filterbank to a rfft signal
  """
  nr = rfft_signal.shape[0]
  subbands = filters[:, :nr] * rfft_signal
  return subbands
