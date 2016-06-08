import scipy
import numpy as np


def freq2erb(freq_hz):
  """ Converts Hz to ERBs, using the formula of Glasberg and Moore.

  Args:
    freq_hz (int, float): frequency to use for ERB.

  Returns:
    n_erb (float)
  """
  return 9.265*np.log(1+freq_hz/(24.7*9.265))


def erb2freq(n_erb):
  """ Converts ERBs to Hz, using the formula of Glasberg and Moore.

  Args:
    n_erb (float)

  Returns:
    freq_hz (int, float)
  """
  return 24.7*9.265*(np.exp(n_erb/9.265)-1)


def make_erb_cos_filts_1x(signal_len, sr, N, low_lim, hi_lim):
  """ Returns N+2 filters as ??column vectors of FILTS

  filters have cosine-shaped frequency responses, with center frequencies
  equally spaced on an ERB scale from low_lim to hi_lim

  Adjacent filters overlap by 50%.

  Args:
    signal_len (int): Length of input signal. Filters are to be applied
      multiplicatively in the frequency domain and thus have a length that
      scales with the signal length (signal_len).
    sr (int): is the sampling rate
    N (int): number of filters to create
    low_lim (int): low cutoff of lowest band
    hi_lim (int): high cutoff of highest band

  Returns:
    filts (np.array): There are N+2 filters because filts also contains lowpass
      and highpass filters to cover the ends of the spectrum.
    hz_cutoffs (np.array): is a vector of the cutoff frequencies of each filter.
      Because of the overlap arrangement, the upper cutoff of one filter is the
      center frequency of its neighbor.
    freqs (np.array): is a vector of frequencies the same length as FILTS, that
      can be used to plot the frequency response of the filters.

  The squared frequency responses of the filters sums to 1, so that they
  can be applied once to generate subbands and then again to collapse the
  subbands to generate a sound signal, without changing the frequency
  content of the signal.

  intended for use with GENERATE_SUBBANDS and COLLAPSE_SUBBANDS
  """
  pass


def runTests():
  # generates filters with cosine frequency response functions on an erb-transformed frequency axis
  SIG_LEN = 1000  #??
  SR = 48000
  N = 30
  LOW_LIM = 100
  HI_LIM =20000
  make_erb_cos_filts_1x(SIG_LEN, SR, N, LOW_LIM, HI_LIM)


if __name__ == '__main__':
  runTests()
