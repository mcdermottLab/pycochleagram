from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np

from pycochleagram import utils


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


def reshape_signal_batch(signal):
  """Convert the signal into a standard batch shape for use with cochleagram.py
  functions. The first dimension is the batch dimension.

  Args:
    signal (array): The sound signal (waveform) in the time domain. Should be
      either a flattened array with shape (n_samples,), a row vector with shape
      (1, n_samples), a column vector with shape (n_samples, 1), or a 2D
      matrix of the form [batch, waveform].

  Returns:
    array:
    **out_signal**: If the input `signal` has a valid shape, returns a
      2D version of the signal with the first dimension as the batch
      dimension.

  Raises:
    ValueError: Raises an error of the input `signal` has invalid shape.
  """
  if signal.ndim == 1:  # signal is a flattened array
    out_signal = signal.reshape((1, -1))
  elif signal.ndim == 2:  # signal is a row or column vector
    if signal.shape[0] == 1:
      out_signal = signal
    elif signal.shape[1] == 1:
      out_signal = signal.reshape((1, -1))
    else:  # first dim is batch dim
      out_signal = signal
  else:
    raise ValueError('signal should be flat array, row or column vector, or a 2D matrix with dimensions [batch, waveform]; found %s' % signal.ndim)
  return out_signal


def generate_subband_envelopes_fast(signal, filters, padding_size=None, fft_mode='auto', debug_ret_all=False):
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
    padding_size (int, optional): Factor that determines if the signal will be
      zero-padded before generating the subbands. If this is None,
      or less than 1, no zero-padding will be used. Otherwise, zeros are added
      to the end of the input signal until is it of length
      `padding_size * length(signal)`. This padded region will be removed after
      performing the subband decomposition.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.

  Returns:
    array:
    **subband_envelopes**: The subband envelopes (i.e., cochleagram) resulting from
      the subband decomposition. This should have the same shape as `filters`.
  """
  # convert the signal to a canonical representation
  signal_flat = reshape_signal_canonical(signal)

  if padding_size is not None and padding_size > 1:
    signal_flat, padding = pad_signal(signal_flat, padding_size)

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

  if padding_size is not None and padding_size > 1:
    analytic_subbands = analytic_subbands[:, :signal_flat.shape[0] - padding]  # i dont know if this is correct
    subband_envelopes = subband_envelopes[:, :signal_flat.shape[0] - padding]  # i dont know if this is correct

  if debug_ret_all is True:
    out_dict = {}
    # add all local variables to out_dict
    for k in dir():
      if k != 'out_dict':
        out_dict[k] = locals()[k]
    return out_dict
  else:
    return subband_envelopes


def generate_subbands(signal, filters, padding_size=None, fft_mode='auto', debug_ret_all=False):
  """Generate the subband decomposition of the signal by applying the provided
  filters.

  The input filters are applied to the signal to perform subband decomposition.
  The signal can be optionally zero-padded before the decomposition.

  Args:
    signal (array): The sound signal (waveform) in the time domain.
    filters (array): The filterbank, in frequency space, used to generate the
      cochleagram. This should be the full filter-set output of
      erbFilter.make_erb_cos_filters_nx, or similar.
    padding_size (int, optional): Factor that determines if the signal will be
      zero-padded before generating the subbands. If this is None,
      or less than 1, no zero-padding will be used. Otherwise, zeros are added
      to the end of the input signal until is it of length
      `padding_size * length(signal)`. This padded region will be removed after
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
  # if padding_size is not None and padding_size >= 1:
  #   padding = signal.shape[0] * padding_size - signal.shape[0]
  #   print('padding ', padding)
  #   signal = np.concatenate((signal, np.zeros(padding)))

  # convert the signal to a canonical representation
  signal_flat = reshape_signal_canonical(signal)

  if padding_size is not None and padding_size > 1:
    signal_flat, padding = pad_signal(signal_flat, padding_size)

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

  if padding_size is not None and padding_size > 1:
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


def generate_analytic_subbands(signal, filters, padding_size=None, fft_mode='auto'):
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
    padding_size (int, optional): Factor that determines if the signal will be zero-padded
      before generating the subbands. If this is None, or less than 1, no
      zero-padding will be used. Otherwise, zeros are added to the end of the
      input signal until is it of length `padding_size * length(signal)`. This
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
  signal_flat = reshape_signal_canonical(signal)

  if padding_size is not None and padding_size > 1:
    signal_flat, padding = pad_signal(signal_flat, padding_size)

  fft_sample = utils.fft(signal_flat, mode=fft_mode)
  subbands = filters * fft_sample
  analytic_subbands = utils.fhilbert(subbands, mode=fft_mode)

  if padding_size is not None and padding_size > 1:
    analytic_subbands = analytic_subbands[:, :signal_flat.shape[0] - padding]  # i dont know if this is correct

  return analytic_subbands


def generate_subband_envelopes(signal, filters, padding_size=None, debug_ret_all=False):
  """Generate the subband envelopes (i.e., the cochleagram) of the signal by
    applying the provided filters.

  The input filters are applied to the signal to perform subband decomposition.
  The signal can be optionally zero-padded before the decomposition.

  Args:
    signal (array): The sound signal (waveform) in the time domain.
    filters (array): The filterbank, in frequency space, used to generate the
      cochleagram. This should be the full filter-set output of
      erbFilter.make_erb_cos_filters_nx, or similar.
    padding_size (int, optional): Factor that determines if the signal will be zero-padded
      before generating the subbands. If this is None, or less than 1, no
      zero-padding will be used. Otherwise, zeros are added to the end of the
      input signal until is it of length `padding_size * length(signal)`. This
      padded region will be removed after performing the subband
      decomposition.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.

  Returns:
    array:
    **subband_envelopes**: The subband envelopes (i.e., cochleagram) resulting from
      the subband decomposition. This should have the same shape as `filters`.
  """
  analytic_subbands = generate_analytic_subbands(signal, filters, padding_size=padding_size)
  subband_envelopes = np.abs(analytic_subbands)

  if debug_ret_all is True:
    out_dict = {}
    # add all local variables to out_dict
    for k in dir():
      if k != 'out_dict':
        out_dict[k] = locals()[k]
    return out_dict
  else:
    return subband_envelopes


def collapse_subbands(subbands, filters, fft_mode='auto'):
  """Collapse the subbands into a waveform by (re)applying the filterbank.

  Args:
    subbands (array): The subband decomposition (i.e., cochleagram) to collapse.
    filters (array): The filterbank, in frequency space, used to generate the
      cochleagram. This should be the full filter-set output of
      erbFilter.make_erb_cos_filters_nx, or similar, that was used to create
      `subbands`.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.

  Returns:
    array:
    **signal**: The signal resulting from collapsing the subbands.
  """
  fft_subbands = filters * utils.fft(subbands, mode=fft_mode)
  # subbands = utils.ifft(fft_subbands)
  subbands = np.real(utils.ifft(fft_subbands, mode=fft_mode))
  signal = subbands.sum(axis=0)
  return signal


def pad_signal(signal, padding_size, axis=0):
  """Pad the signal by appending zeros to the end. The padded signal has
  length `padding_size * length(signal)`.

  Args:
    signal (array): The signal to be zero-padded.
    padding_size (int): Factor that determines the size of the padded signal.
      The padded signal has length `padding_size * length(signal)`.
    axis (int): Specifies the axis to pad; defaults to 0.

  Returns:
    tuple:
      **pad_signal** (*array*): The zero-padded signal.
      **padding_size** (*int*): The length of the zero-padding added to the array.
  """
  if padding_size is not None and padding_size >= 1:
    pad_shape = list(signal.shape)
    pad_shape[axis] = padding_size
    pad_signal = np.concatenate((signal, np.zeros(pad_shape)))
  else:
    padding_size = 0
    pad_signal = signal
  return (pad_signal, padding_size)


def _real_freq_filter(rfft_signal, filters):
  """Helper function to apply a full filterbank to a rfft signal
  """
  nr = rfft_signal.shape[0]
  subbands = filters[:, :nr] * rfft_signal
  return subbands
