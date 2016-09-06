from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import math
import scipy
import numpy as np
import matplotlib.pyplot as plt

import utils

import pdb


def freq2erb(freq_hz):
  """ Converts Hz to ERBs, using the formula of Glasberg and Moore.

  Args:
    freq_hz (int, float): frequency to use for ERB.

  Returns:
    n_erb (float)
  """
  return 9.265 * np.log(1 + freq_hz / (24.7 * 9.265))


def erb2freq(n_erb):
  """ Converts ERBs to Hz, using the formula of Glasberg and Moore.

  Args:
    n_erb (int, float)

  Returns:
    freq_hz (float)
  """
  return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1)


def make_cosine_filter(freqs, l, h, convert_to_erb=True):
  if convert_to_erb:
    freqs_erb = freq2erb(freqs)
    l_erb = freq2erb(l)
    h_erb = freq2erb(h)

  avg_in_erb = (l_erb + h_erb) / 2  # center of filter
  rnge_in_erb = h_erb - l_erb  # width of filter
  # return np.cos((freq2erb(freqs[a_l_ind:a_h_ind+1]) - avg)/rnge * np.pi)  # h_ind+1 to include endpoint
  return np.cos((freqs_erb[(freqs_erb > l_erb) & (freqs_erb < h_erb)]- avg_in_erb) / rnge_in_erb * np.pi)  # map cutoffs to -pi/2, pi/2 interval


def make_erb_cos_filters_nx(signal_length, sr, n, low_lim, hi_lim, sample_factor, strict=True):
  """ Create ERB cosine filters, oversampled by a factor provided by "sample_factor"
  """
  if not isinstance(sample_factor, int):
    raise ValueError('sample_factor must be an integer, not %s' % type(sample_factor))
  if sample_factor <= 0:
    raise ValueError('sample_factor must be positive')

  if sample_factor != 1 and np.remainder(sample_factor, 2) != 0:
    if strict:
      msg = 'sample_factor odd, and will change ERB filter widths. Use even sample factors for comparison.'
      raise ValueError(msg)
    else:
      warnings.warn(msg, RuntimeWarning, stacklevel=2)

  if np.remainder(signal_length, 2) == 0:  # even length
    n_freqs = signal_length / 2  # .0 does not include DC, likely the sampling grid
    max_freq = sr / 2  # go all the way to nyquist
  else:  # odd length
    n_freqs = (signal_length - 1) / 2  # .0
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
  cos_filts = np.zeros((n_freqs + 1 , n_filters + n_lp_hp))  # ?? n_freqs+1

  # cutoffs are evenly spaced on an erb scale -- interpolate linearly in erb space then convert back
  # erb_spacing = (freq2erb(hi_lim) - freq2erb(low_lim)) / (n_filters + 1)
  # center_freqs = np.linspace(freq2erb(low_lim) + erb_spacing, freq2erb(hi_lim) - erb_spacing, n_filters)
  # get the actual spacing use to generate the sequence (in case numpy does something weird)
  center_freqs, erb_spacing = np.linspace(freq2erb(low_lim), freq2erb(hi_lim), n_filters + 2, retstep=True)  # +2 for bin endpoints
  # we need to exclude the endpoints
  center_freqs = center_freqs[1:-1]
  # assert erb_spacing == erb_spacing0, 'ERB spacing mismatch'

  print(center_freqs)
  print('sample_factor: ', sample_factor)
  for i in range(n_filters):
    i_offset = i + sample_factor
    l = erb2freq(center_freqs[i] - sample_factor * erb_spacing)
    h = erb2freq(center_freqs[i] + sample_factor * erb_spacing)
    print('i: %s, l, h: [%s, %s]' % (i, l, h))
    ## Ray 2 ##
    temp_cos_filts = make_cosine_filter(freqs, l, h)
    print(temp_cos_filts.shape)
    # the first sample_factor # of rows in cos_filts will be lowpass filters
    cos_filts[(freqs > l) & (freqs < h), i_offset] = temp_cos_filts


  # make lowpass and highpass filters for perfect reconstruction
  # there should be sample_factor number of both low-pass and high pass filters
  # lp_filts = np.zeros((cos_filts.shape[0], sample_factor))
  # hp_filts = np.zeros((cos_filts.shape[0], sample_factor))
  for i in range(sample_factor):
    # account for the fact that the first sample_factor # of filts are lowpass
    i_offset = i + sample_factor
    print('lp/hp inds: [%s, %s]' % (i_offset, -1-i_offset))
    lp_h_ind = max(np.where(freqs < erb2freq(center_freqs[0]))[0])  # lowpass filter goes up to peak of first cos filter
    lp_filt = np.sqrt( 1 - np.power(cos_filts[:lp_h_ind+1, i_offset], 2))
    # lp_filts[:lp_h_ind+1, i] = lp_filt

    hp_l_ind = min(np.where(freqs > erb2freq(center_freqs[-1-i]))[0])  # highpass filter goes down to peak of last cos filter
    hp_filt = np.sqrt(1 - np.power(cos_filts[hp_l_ind:, -1-i_offset], 2))
    # hp_filts[hp_l_ind:, -1-i] = hp_filt

    cos_filts[:lp_h_ind+1, i] = lp_filt
    cos_filts[hp_l_ind:, -1-i] = hp_filt

  # cos_filts = np.hstack((lp_filts, cos_filts, hp_filts))

  cos_filts = cos_filts / np.sqrt(sample_factor)  # so that squared freq response adds to one

  # get center freqs for lowpass and highpass filters
  cfs_low = np.copy(center_freqs[:sample_factor]) - sample_factor * erb_spacing
  cfs_hi = np.copy(center_freqs[-sample_factor:]) + sample_factor * erb_spacing
  center_freqs = erb2freq(np.concatenate((cfs_low, center_freqs, cfs_hi)))

  # rectify
  # center_freqs[center_freqs < 0] = 1

  # flip thing

  plt.plot(cos_filts)
  plt.show()

  return cos_filts, center_freqs, freqs


def make_erb_cos_filters_1x(signal_length, sr, low_lim, hi_lim, strict=True):
  return make_erb_cos_filters_nx(signal_length, sr, 1, low_lim, hi_lim, strict=strict)


def make_erb_cos_filters_2x(signal_length, sr, low_lim, hi_lim, strict=True):
  return make_erb_cos_filters_nx(signal_length, sr, 2, low_lim, hi_lim, strict=strict)


def make_erb_cos_filters_4x(signal_length, sr, low_lim, hi_lim, strict=True):
  return make_erb_cos_filters_nx(signal_length, sr, 4, low_lim, hi_lim, strict=strict)


def make_erb_cos_filters(signal_length, sr, n, low_lim, hi_lim, strict=True):
  """ Returns n+2 filters as ??column vectors of FILTS

  filters have cosine-shaped frequency responses, with center frequencies
  equally spaced on an ERB scale from low_lim to hi_lim

  Adjacent filters overlap by 50%.

  Args:
    signal_length (int): Length of input signal. Filters are to be applied
      multiplicatively in the frequency domain and thus have a length that
      scales with the signal length (signal_length).
    sr (int): is the sampling rate
    n (int): number of filters to create
    low_lim (int): low cutoff of lowest band
    hi_lim (int): high cutoff of highest band

  Returns:
    filts (np.array): There are n+2 filters because filts also contains lowpass
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


  # make cutoffs evenly spaced on an erb scale
  # cutoffs = erb2freq([freq2erb(low_lim) : (freq2erb(hi_lim)-freq2erb(low_lim))/(n+1) : freq2erb(hi_lim)])

  # # cutoffs are evenly spaced on an erb scale -- interpolate linearly in erb space then convert back
  # erb_spacing = (freq2erb(hi_lim)-freq2erb(low_lim))/(n+1)
  # center_freqs = np.linspace(freq2erb(low_lim)+erb_spacing, freq2erb(hi_lim)-erb_spacing, n)
  # cutoffs = erb2freq(center_freqs)

  # cutoffs are evenly spaced on an erb scale -- interpolate linearly in erb space then convert back
  # cutoffs_in_erb = np.linspace(freq2erb(low_lim), freq2erb(hi_lim), n+2)
  cutoffs_in_erb = utils.matlab_arange(freq2erb(low_lim), freq2erb(hi_lim), n + 1)  # ?? n+1
  cutoffs = erb2freq(cutoffs_in_erb)

  plt.figure()
  plt.show(block=False)
  ax = plt.gca()
  # generate cosine filters
  for k in range(n):
    l = cutoffs[k]
    h = cutoffs[k + 2]  # adjacent filters overlap by 50%
    # print(l, h)
    # print(freqs.shape)
    l_ind = min(np.where(freqs > l)[0])
    h_ind = max(np.where(freqs < h)[0])
    print('RAY: ', l_ind, ' | ', h_ind)

    a_l_ind = np.nonzero(freqs > l)[0][0]  # hacky min
    a_h_ind = np.nonzero(freqs < h)[-1][-1]  # hacky max
    print('ALEX: ', a_l_ind, ' | ', a_h_ind)
    assert a_l_ind == l_ind
    assert a_h_ind == h_ind

    # ax.plot(test_freqs)
    # plt.draw()
    # plt.waitforbuttonpress()

    # if k == 5:
    #   exit()

    avg = (freq2erb(l) + freq2erb(h)) / 2  # center of filter?
    rnge = freq2erb(h) - freq2erb(l)  # width of filter

    # cos_filts[(freqs > l) & (freqs < h), k] = np.cos( (freq2erb(freqs[(freqs > l) & (freqs < h)]) - avg)/rnge * np.pi)
    # xx = np.cos( (freq2erb(freqs[(freqs > l) & (freqs < h)]) - avg)/rnge * np.pi)
    xx = make_cosine_filter(freqs, l, h)
    cos_filts[(freqs > l) & (freqs < h), k] = xx

    # a_cos_filts[a_l_ind:a_h_ind+1,k] = np.cos( (freq2erb(freqs[a_l_ind:a_h_ind+1]) - avg)/rnge * np.pi)
    axx = np.cos((freq2erb(freqs[a_l_ind:a_h_ind+1]) - avg)/rnge * np.pi)  # h_ind+1 to include endpoint
    a_cos_filts[a_l_ind:a_h_ind+1,k] = axx
    print(":CAT", a_cos_filts[a_l_ind:a_h_ind+1,k].shape)

    # ax.plot(xx, c='r', marker='D')
    # plt.draw()
    # ax.plot(axx, c='b', marker='+')
    # plt.draw()
    # plt.waitforbuttonpress()
    assert np.all(xx == axx)

  print(cos_filts.shape)

  # add lowpass and highpass for perfect reconstruction
  filts = np.zeros((n_freqs + 1, n + 2))
  filts[:,1:n+1] = cos_filts
  print(cos_filts[:,:1].shape)
  lp_filt = np.zeros_like(cos_filts[:, :0])
  hp_filt = np.copy(lp_filt)

  h_ind = max(np.where(freqs < cutoffs[1])[0])  # lowpass filter goes up to peak of first cos filter
  l_ind = min(np.where(freqs > cutoffs[n])[0])  # lowpass filter goes up to peak of first cos filter
  a_h_ind = np.nonzero(freqs<cutoffs[1])[-1][-1]  # hacky max
  a_l_ind = np.nonzero(freqs>cutoffs[n])[0][0] # hacky min
  assert(a_h_ind == h_ind)
  assert(a_l_ind == l_ind)

  # filts[:h_ind+1,0] = np.sqrt( 1 - filts[:h_ind+1,1]**2)
  filts[:h_ind+1, 0] = np.sqrt(1 - np.power(filts[:h_ind+1,1], 2.0))
  a = freqs > cutoffs[n+1]
  print('freqs shape', freqs.shape)
  print('a shape', a.shape)
  print('cos filts shape', cos_filts.shape)
  print('filts shape', filts.shape)
  yy = np.sqrt(1.0 - cos_filts[freqs > cutoffs[n], -1]**2.0)
  # plt.plot(yy, c='r')
  filts[l_ind:n_freqs+2,n+1] = np.sqrt(1 - np.power(filts[l_ind:n_freqs+2,n], 2))

  # plt.plot(filts)
  # plt.plot(filts[l_ind:n_freqs+2,n+1], c='k')
  # plt.plot(yy, c='k')
  # plt.plot(filts[:, 0], c='b')
  # plt.draw()
  # plt.waitforbuttonpress()
  # plt.waitforbuttonpress()

  filts = np.hstack((lp_filt, cos_filts, hp_filt))
  print(filts.shape)


  # add lowpass and highpass for perfect reconstruction
  filts = np.zeros((n_freqs+1,n+2))
  filts[:,1:n+1] = cos_filts
  h_ind = np.nonzero(freqs<cutoffs[1])[-1][-1] # hacky max
  filts[:h_ind+1,0] = np.sqrt( 1 - filts[:h_ind+1,1]**2)
  l_ind = np.nonzero(freqs>cutoffs[n])[0][0] # hacky min
  filts[l_ind:n_freqs+2,n+1] = np.sqrt(1.0 - filts[l_ind:n_freqs+2,n]**2.0)
  plt.plot(filts[l_ind:n_freqs+2,n+1])
  plt.draw()
  plt.show()

  print('filts shpae', filts.shape[1])
  for i in range(filts.shape[1]):
    print('plotting ',i)
    plt.plot(filts[:, i])
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

  return filts, cutoffs, freqs


def runTests():
  import matplotlib.pyplot as plt
  # generates filters with cosine frequency response functions on an erb-transformed frequency axis
  SIG_LEN = 1000  # ??
  SR = 44100
  LOW_LIM = 50
  HI_LIM = 20000
  N_HUMAN = np.floor(freq2erb(HI_LIM) - freq2erb(LOW_LIM)) - 1;
  N_HUMAN = N_HUMAN.astype(int)
  N_HUMAN = 3
  print('N: ', N_HUMAN)

  t = utils.matlab_arange(0, 1, SR)
  ct = np.sin(2*np.pi*10*t)

  print(t.shape)

  # alex_makeErbCosFilts1x(len(ct), SR, n, LOW_LIM, HI_LIM)
  # make_erb_cos_filters_1x(len(ct), SR, N_HUMAN, LOW_LIM, HI_LIM)
  # make_erb_cos_filters_2x(len(ct), SR, N_HUMAN, LOW_LIM, HI_LIM)
  plt.figure()
  filts, hz_cutoffs, freqs = make_erb_cos_filters_nx(len(ct), SR, N_HUMAN, LOW_LIM, HI_LIM, 1)
  filts, hz_cutoffs, freqs = make_erb_cos_filters_nx(len(ct), SR, N_HUMAN, LOW_LIM, HI_LIM, 2)
  filts, hz_cutoffs, freqs = make_erb_cos_filters_nx(len(ct), SR, N_HUMAN, LOW_LIM, HI_LIM, 4)
  exit()


if __name__ == '__main__':
  runTests()
