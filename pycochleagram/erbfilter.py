from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np

from pycochleagram import utils


def freq2erb(freq_hz):
  """Converts Hz to human-defined ERBs, using the formula of Glasberg and Moore.

  Args:
    freq_hz (int, float): frequency to use for ERB.

  Returns:
    float:
    **n_erb**: Human-defined ERB representation of input.
  """
  return 9.265 * np.log(1 + freq_hz / (24.7 * 9.265))


def erb2freq(n_erb):
  """Converts human ERBs to Hz, using the formula of Glasberg and Moore.

  Args:
    n_erb (int, float)

  Returns:
    float:
    **freq_hz**: Frequency representation of input
  """
  return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1)


def make_cosine_filter(freqs, l, h, convert_to_erb=True):
  """Generate a half-cosine filter. Represents one subband of the cochleagram.

  A half-cosine filter is created using the values of freqs that are within the
  interval [l, h]. The half-cosine filter is centered at the center of this
  interval, i.e., (h - l) / 2. Values outside the valid interval [l, h] are
  discarded. So, if freqs = [1, 2, 3, ... 10], l = 4.5, h = 8, the cosine filter
  will only be defined on the domain [5, 6, 7] and the returned output will only
  contain 3 elements.

  Args:
    freqs (array): Array containing the domain of the filter, in ERB space;
      see convert_to_erb parameter below.. A single half-cosine
      filter will be defined only on the valid section of these values;
      specifically, the values between cutoffs "l" and "h". A half-cosine filter
      centered at (h - l ) / 2 is created on the interval [l, h].
    l (number): The lower cutoff of the half-cosine filter in ERB space; see
      convert_to_erb parameter below.
    h (number): The upper cutoff of the half-cosine filter in ERB space; see
      convert_to_erb parameter below.
    convert_to_erb (bool, optional): If this is True (default), the values in
      input arguments "freqs", "l", and "h" will be transformed from Hz to ERB
      space before creating the half-cosine filter. If this is False, the
      input arguments are assumed to be in ERB space.

  Returns:
    array:
    **half_cos_filter**: A half-cosine filter defined using elements of
      freqs within [l, h].
  """
  if convert_to_erb:
    freqs_erb = freq2erb(freqs)
    l_erb = freq2erb(l)
    h_erb = freq2erb(h)
  else:
    freqs_erb = freqs
    l_erb = l
    h_erb = h

  avg_in_erb = (l_erb + h_erb) / 2  # center of filter
  rnge_in_erb = h_erb - l_erb  # width of filter
  # return np.cos((freq2erb(freqs[a_l_ind:a_h_ind+1]) - avg)/rnge * np.pi)  # h_ind+1 to include endpoint
  # return np.cos((freqs_erb[(freqs_erb >= l_erb) & (freqs_erb <= h_erb)]- avg_in_erb) / rnge_in_erb * np.pi)  # map cutoffs to -pi/2, pi/2 interval
  return np.cos((freqs_erb[(freqs_erb > l_erb) & (freqs_erb < h_erb)]- avg_in_erb) / rnge_in_erb * np.pi)  # map cutoffs to -pi/2, pi/2 interval


def make_full_filter_set(filts, signal_length=None):
  """Create the full set of filters by extending the filterbank to negative FFT
  frequencies.

  Args:
    filts (array): Array containing the cochlear filterbank in frequency space,
      i.e., the output of make_erb_cos_filters_nx. Each row of filts is a
      single filter, with columns indexing frequency.
    signal_length (int, optional): Length of the signal to be filtered with this filterbank.
      This should be equal to filter length * 2 - 1, i.e., 2*filts.shape[1] - 1, and if
      signal_length is None, this value will be computed with the above formula.
      This parameter might be deprecated later.

  Returns:
    array:
    **full_filter_set**: Array containing the complete filterbank in
      frequency space. This output can be directly applied to the frequency
      representation of a signal.
  """
  if signal_length is None:
    signal_length = 2 * filts.shape[1] - 1

  # note that filters are currently such that each ROW is a filter and COLUMN idxs freq
  if np.remainder(signal_length, 2) == 0:  # even -- don't take the DC & don't double sample nyquist
    neg_filts = np.flipud(filts[1:filts.shape[0] - 1, :])
  else:  # odd -- don't take the DC
    neg_filts = np.flipud(filts[1:filts.shape[0], :])
  fft_filts = np.vstack((filts, neg_filts))
  # we need to switch representation to apply filters to fft of the signal, not sure why, but do it here
  return fft_filts.T


def make_erb_cos_filters_nx(signal_length, sr, n, low_lim, hi_lim, sample_factor, padding_size=None, full_filter=True, strict=True, **kwargs):
  """Create ERB cosine filters, oversampled by a factor provided by "sample_factor"

  Args:
    signal_length (int): Length of signal to be filtered with the generated
      filterbank. The signal length determines the length of the filters.
    sr (int): Sampling rate associated with the signal waveform.
    n (int): Number of filters (subbands) to be generated with standard
      sampling (i.e., using a sampling factor of 1). Note, the actual number of
      filters in the generated filterbank depends on the sampling factor, and
      will also include lowpass and highpass filters that allow for
      perfect reconstruction of the input signal (the exact number of lowpass
      and highpass filters is determined by the sampling factor). The
      number of filters in the generated filterbank is given below:
      ```
        sample factor  |    n_out      |=|  bandpass  |+|  highpass + lowpass
        ---------------|-------------- |=|------------|+|--------------------
              1        |     n+2       |=|     n      |+|      1    +    1
              2        |   2*n+1+4     |=|   2*n+1    |+|      2    +    2
              4        |   4*n+3+8     |=|   4*n+3    |+|      4    +    4
              s        | s*(n+1)-1+2*s |=|  s*(n+1)-1 |+|      s    +    s
      ```
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
      waveform signal will be padded to length padding_size*signal_length.
    full_filter (bool, optional): If True (default), the complete filter that
      is ready to apply to the signal is returned. If False, only the first
      half of the filter is returned (likely positive terms of FFT).
    strict (bool, optional): If True (default), will throw an error if
      sample_factor is not a power of two. This facilitates comparison across
      sample_factors. Also, if True, will throw an error if provided hi_lim
      is greater than the Nyquist rate.

  Returns:
      tuple:
        **filts** (*array*): The filterbank consisting of filters have
          cosine-shaped frequency responses, with center frequencies equally
          spaced on an ERB scale from low_lim to hi_lim.
        **center_freqs** (*array*):
        **freqs** (*array*):

  Raises:
      ValueError: Various value errors for bad choices of sample_factor; see
        description for strict parameter.
  """
  if not isinstance(sample_factor, int):
    raise ValueError('sample_factor must be an integer, not %s' % type(sample_factor))
  if sample_factor <= 0:
    raise ValueError('sample_factor must be positive')

  if sample_factor != 1 and np.remainder(sample_factor, 2) != 0:
    msg = 'sample_factor odd, and will change ERB filter widths. Use even sample factors for comparison.'
    if strict:
      raise ValueError(msg)
    else:
      warnings.warn(msg, RuntimeWarning, stacklevel=2)

  if padding_size is not None and padding_size >= 1:
    signal_length += padding_size

  if np.remainder(signal_length, 2) == 0:  # even length
    n_freqs = signal_length // 2  # .0 does not include DC, likely the sampling grid
    max_freq = sr / 2  # go all the way to nyquist
  else:  # odd length
    n_freqs = (signal_length - 1) // 2  # .0
    max_freq = sr * (signal_length - 1) / 2 / signal_length  # just under nyquist

  # verify the high limit is allowed by the sampling rate
  if hi_lim > sr / 2:
    hi_lim = max_freq
    msg = 'input arg "hi_lim" exceeds nyquist limit for max frequency; ignore with "strict=False"'
    if strict:
      raise ValueError(msg)
    else:
      warnings.warn(msg, RuntimeWarning, stacklevel=2)

  # changing the sampling density without changing the filter locations
  # (and, thereby changing their widths) requires that a certain number of filters
  # be used.
  n_filters = sample_factor * (n + 1) - 1
  n_lp_hp = 2 * sample_factor
  freqs = utils.matlab_arange(0, max_freq, n_freqs)
  filts = np.zeros((n_freqs + 1 , n_filters + n_lp_hp))  # ?? n_freqs+1

  # cutoffs are evenly spaced on an erb scale -- interpolate linearly in erb space then convert back
  # get the actual spacing use to generate the sequence (in case numpy does something weird)
  center_freqs, erb_spacing = np.linspace(freq2erb(low_lim), freq2erb(hi_lim), n_filters + 2, retstep=True)  # +2 for bin endpoints
  # we need to exclude the endpoints
  center_freqs = center_freqs[1:-1]

  freqs_erb = freq2erb(freqs)
  for i in range(n_filters):
    i_offset = i + sample_factor
    l = center_freqs[i] - sample_factor * erb_spacing
    h = center_freqs[i] + sample_factor * erb_spacing
    # the first sample_factor # of rows in filts will be lowpass filters
    filts[(freqs_erb > l) & (freqs_erb < h), i_offset] = make_cosine_filter(freqs_erb, l, h, convert_to_erb=False)

  # be sample_factor number of each
  for i in range(sample_factor):
    # account for the fact that the first sample_factor # of filts are lowpass
    i_offset = i + sample_factor
    lp_h_ind = max(np.where(freqs < erb2freq(center_freqs[i]))[0])  # lowpass filter goes up to peak of first cos filter
    lp_filt = np.sqrt(1 - np.power(filts[:lp_h_ind+1, i_offset], 2))

    hp_l_ind = min(np.where(freqs > erb2freq(center_freqs[-1-i]))[0])  # highpass filter goes down to peak of last cos filter
    hp_filt = np.sqrt(1 - np.power(filts[hp_l_ind:, -1-i_offset], 2))

    filts[:lp_h_ind+1, i] = lp_filt
    filts[hp_l_ind:, -1-i] = hp_filt

  # ensure that squared freq response adds to one
  filts = filts / np.sqrt(sample_factor)

  # get center freqs for lowpass and highpass filters
  cfs_low = np.copy(center_freqs[:sample_factor]) - sample_factor * erb_spacing
  cfs_hi = np.copy(center_freqs[-sample_factor:]) + sample_factor * erb_spacing
  center_freqs = erb2freq(np.concatenate((cfs_low, center_freqs, cfs_hi)))

  # rectify
  center_freqs[center_freqs < 0] = 1

  # discard highpass and lowpass filters, if requested
  if kwargs.get('no_lowpass'):
    filts = filts[:, sample_factor:]
  if kwargs.get('no_highpass'):
    filts = filts[:, :-sample_factor]

  # make the full filter by adding negative components
  if full_filter:
    filts = make_full_filter_set(filts, signal_length)

  return filts, center_freqs, freqs


def make_erb_cos_filters_1x(signal_length, sr, n, low_lim, hi_lim, padding_size=None, full_filter=False, strict=False):
  """Create ERB cosine filterbank, sampled from ERB at 1x overcomplete.

  Returns n+2 filters as ??column vector

  filters have cosine-shaped frequency responses, with center frequencies
  equally spaced on an ERB scale from low_lim to hi_lim

  Adjacent filters overlap by 50%.

  The squared frequency responses of the filters sums to 1, so that they
  can be applied once to generate subbands and then again to collapse the
  subbands to generate a sound signal, without changing the frequency
  content of the signal.

  intended for use with GENERATE_SUBBANDS and COLLAPSE_SUBBANDS

  Args:
    signal_length (int): Length of input signal. Filters are to be applied
      multiplicatively in the frequency domain and thus have a length that
      scales with the signal length (signal_length).
    sr (int): is the sampling rate
    n (int): number of filters to create
    low_lim (int): low cutoff of lowest band
    hi_lim (int): high cutoff of highest band
    padding_size (int, optional): If None (default), the signal will not be padded
      before filtering. Otherwise, the filters will be created assuming the
      waveform signal will be padded to length padding_size*signal_length.
    full_filter (bool, optional): If True, the complete filter that
      is ready to apply to the signal is returned. If False (default), only the first
      half of the filter is returned (likely positive terms of FFT).
    strict (bool, optional): If True (default), will throw an error if provided
      hi_lim is greater than the Nyquist rate.

  Returns:
    tuple:
      **filts** (*array*): There are n+2 filters because filts also contains lowpass
        and highpass filters to cover the ends of the spectrum.
      **hz_cutoffs** (*array*): is a vector of the cutoff frequencies of each filter.
        Because of the overlap arrangement, the upper cutoff of one filter is the
        center frequency of its neighbor.
      **freqs** (*array*): is a vector of frequencies the same length as filts, that
        can be used to plot the frequency response of the filters.
  """
  return make_erb_cos_filters_nx(signal_length, sr, n, low_lim, hi_lim, 1, padding_size=padding_size, full_filter=full_filter, strict=strict)


def make_erb_cos_filters_2x(signal_length, sr, n, low_lim, hi_lim, padding_size=None, full_filter=False, strict=False):
  """Create ERB cosine filterbank, sampled from ERB at 2x overcomplete.

  Returns 2*n+5 filters as column vectors
  filters have cosine-shaped frequency responses, with center frequencies
  equally spaced on an ERB scale from low_lim to hi_lim

  This function returns a filterbank that is 2x overcomplete compared to
  make_erb_cos_filts_1x (to get filterbanks that can be compared with each
  other, use the same value of n in both cases). Adjacent filters overlap
  by 75%.

  The squared frequency responses of the filters sums to 1, so that they
  can be applied once to generate subbands and then again to collapse the
  subbands to generate a sound signal, without changing the frequency
  content of the signal.

  intended for use with GENERATE_SUBBANDS and COLLAPSE_SUBBANDS

  Args:
    signal_length (int): Length of input signal. Filters are to be applied
      multiplicatively in the frequency domain and thus have a length that
      scales with the signal length (signal_length).
    sr (int): the sampling rate
    n (int): number of filters to create
    low_lim (int): low cutoff of lowest band
    hi_lim (int): high cutoff of highest band
    padding_size (int, optional): If None (default), the signal will not be padded
      before filtering. Otherwise, the filters will be created assuming the
      waveform signal will be padded to length padding_size*signal_length.
    full_filter (bool, optional): If True, the complete filter that
      is ready to apply to the signal is returned. If False (default), only the first
      half of the filter is returned (likely positive terms of FFT).
    strict (bool, optional): If True, will throw an error if provided hi_lim
      is greater than the Nyquist rate.

  Returns:
    tuple: tuple containing:
      **filts** (*array*): There are 2*n+5 filters because filts also contains lowpass
        and highpass filters to cover the ends of the spectrum and sampling
        is 2x overcomplete.
      **hz_cutoffs** (*array*): is a vector of the cutoff frequencies of each filter.
        Because of the overlap arrangement, the upper cutoff of one filter is the
        center frequency of its neighbor.
      **freqs** (*array*): is a vector of frequencies the same length as filts, that
        can be used to plot the frequency response of the filters.
  """
  return make_erb_cos_filters_nx(signal_length, sr, n, low_lim, hi_lim, 2, padding_size=padding_size, full_filter=full_filter, strict=strict)


def make_erb_cos_filters_4x(signal_length, sr, n, low_lim, hi_lim, padding_size=None, full_filter=False, strict=False):
  """Create ERB cosine filterbank, sampled from ERB at 4x overcomplete.

  Returns 4*n+11 filters as column vectors
  filters have cosine-shaped frequency responses, with center frequencies
  equally spaced on an ERB scale from low_lim to hi_lim

  This function returns a filterbank that is 4x overcomplete compared to
  MAKE_ERB_COS_FILTS (to get filterbanks that can be compared with each
  other, use the same value of n in both cases). Adjacent filters overlap
  by 87.5%.

  The squared frequency responses of the filters sums to 1, so that they
  can be applied once to generate subbands and then again to collapse the
  subbands to generate a sound signal, without changing the frequency
  content of the signal.

  intended for use with GENERATE_SUBBANDS and COLLAPSE_SUBBANDS

  Args:
    signal_length (int): Length of input signal. Filters are to be applied
      multiplicatively in the frequency domain and thus have a length that
      scales with the signal length (signal_length).
    sr (int): the sampling rate
    n (int): number of filters to create
    low_lim (int): low cutoff of lowest band
    hi_lim (int): high cutoff of highest band
    padding_size (int, optional): If None (default), the signal will not be padded
      before filtering. Otherwise, the filters will be created assuming the
      waveform signal will be padded to length padding_size*signal_length.
    full_filter (bool, optional): If True, the complete filter that
      is ready to apply to the signal is returned. If False (default), only the first
      half of the filter is returned (likely positive terms of FFT).
    strict (bool, optional): If True, will throw an error if provided hi_lim
      is greater than the Nyquist rate.

  Returns:
    tuple:
      **filts** (*array*): There are 4*n+11 filters because filts also contains lowpass
        and highpass filters to cover the ends of the spectrum and sampling
        is 4x overcomplete.
      **hz_cutoffs** (*array*): is a vector of the cutoff frequencies of each filter.
        Because of the overlap arrangement, the upper cutoff of one filter is the
        center frequency of its neighbor.
      **freqs** (*array*): is a vector of frequencies the same length as filts, that
        can be used to plot the frequency response of the filters.
  """
  return make_erb_cos_filters_nx(signal_length, sr, n, low_lim, hi_lim, 4, padding_size=padding_size, full_filter=full_filter, strict=strict)


def make_erb_cos_filters(signal_length, sr, n, low_lim, hi_lim, full_filter=False, strict=False):
  """Fairly literal port of Josh McDermott's MATLAB make_erb_cos_filters. Useful
  for debugging, but isn't very generalizable. Use make_erb_cos_filters_1x or
  make_erb_cos_filters_nx with sample_factor=1 instead.

  Returns n+2 filters as ??column vectors of FILTS

  filters have cosine-shaped frequency responses, with center frequencies
  equally spaced on an ERB scale from low_lim to hi_lim

  Adjacent filters overlap by 50%.

  The squared frequency responses of the filters sums to 1, so that they
  can be applied once to generate subbands and then again to collapse the
  subbands to generate a sound signal, without changing the frequency
  content of the signal.

  intended for use with GENERATE_SUBBANDS and COLLAPSE_SUBBANDS

  Args:
    signal_length (int): Length of input signal. Filters are to be applied
      multiplicatively in the frequency domain and thus have a length that
      scales with the signal length (signal_length).
    sr (int): is the sampling rate
    n (int): number of filters to create
    low_lim (int): low cutoff of lowest band
    hi_lim (int): high cutoff of highest band

  Returns:
    tuple:
      **filts** (*array*): There are n+2 filters because filts also contains lowpass
        and highpass filters to cover the ends of the spectrum.
      **hz_cutoffs** (*array*): is a vector of the cutoff frequencies of each filter.
        Because of the overlap arrangement, the upper cutoff of one filter is the
        center frequency of its neighbor.
      **freqs** (*array*): is a vector of frequencies the same length as filts, that
        can be used to plot the frequency response of the filters.
  """
  if np.remainder(signal_length, 2) == 0:  # even length
    n_freqs = signal_length / 2  # .0 does not include DC, likely the sampling grid
    max_freq = sr / 2  # go all the way to nyquist
  else:  # odd length
    n_freqs = (signal_length - 1) / 2  # .0
    max_freq = sr * (signal_length - 1) / 2 / signal_length  # just under nyquist

  freqs =  utils.matlab_arange(0, max_freq, n_freqs)
  cos_filts = np.zeros((n_freqs + 1, n))  # ?? n_freqs+1
  a_cos_filts = np.zeros((n_freqs+1, n))  # ?? n_freqs+1

  if hi_lim > sr / 2:
    hi_lim = max_freq
    if strict:
      raise ValueError('input arg "hi_lim" exceeds nyquist limit for max '
                       'frequency ignore with "strict=False"')

  # cutoffs are evenly spaced on an erb scale -- interpolate linearly in erb space then convert back
  cutoffs_in_erb = utils.matlab_arange(freq2erb(low_lim), freq2erb(hi_lim), n + 1)  # ?? n+1
  cutoffs = erb2freq(cutoffs_in_erb)

  # generate cosine filters
  for k in range(n):
    l = cutoffs[k]
    h = cutoffs[k + 2]  # adjacent filters overlap by 50%
    l_ind = min(np.where(freqs > l)[0])
    h_ind = max(np.where(freqs < h)[0])
    avg = (freq2erb(l) + freq2erb(h)) / 2  # center of filter
    rnge = freq2erb(h) - freq2erb(l)  # width of filter
    cos_filts[l_ind:h_ind+1,k] = np.cos((freq2erb(freqs[l_ind:h_ind+1]) - avg)/rnge * np.pi)  # h_ind+1 to include endpoint

  # add lowpass and highpass for perfect reconstruction
  filts = np.zeros((n_freqs + 1, n + 2))
  filts[:,1:n+1] = cos_filts
  lp_filt = np.zeros_like(cos_filts[:, :0])
  hp_filt = np.copy(lp_filt)

  # add lowpass and highpass for perfect reconstruction
  filts = np.zeros((n_freqs+1,n+2))
  filts[:,1:n+1] = cos_filts
  h_ind = max(np.where(freqs < cutoffs[1])[0])  # lowpass filter goes up to peak of first cos filter
  filts[:h_ind+1,0] = np.sqrt( 1 - filts[:h_ind+1,1]**2)
  l_ind = min(np.where(freqs > cutoffs[n])[0])  # lowpass filter goes up to peak of first cos filter
  filts[l_ind:n_freqs+2,n+1] = np.sqrt(1.0 - filts[l_ind:n_freqs+2,n]**2.0)

  # make the full filter by adding negative components
  if full_filter:
    filts = make_full_filter_set(filts, signal_length)

  return filts, cutoffs, freqs
