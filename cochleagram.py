from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from erbFilter import *
import utils

import pdb


def generate_subband_envelopes_fast(signal, filters, pad_factor=None, downsample=None, nonlinearity=None):
    """
    generate subbands for a given signal [np array]
    -- uses the filters specified with self.makeErbCosFilters()

    NOTE: as of december 2015 this is a novel chunk of code that expedites
      the calculation of the subbands

    also, this method only returns the envelopes
      the previous function generateSubbands defaulted to not return envelopes,
      so here i'm making the user explicitly specify do_return_envs=True
      -- i've set the default arg to None which will through an error

    the method of speed up consists of using rfft rather than vanilla fft
      to compute the dft.  moreover, i'm implementing the Hilbert transform
      myself since i'm already in the frequency domain -- there's no need to
      back to the time domain

      fourier-based analytic signal algo from here:
      Marple.  Computing the Discrete-Time Analytic Signal via FFT.  IEEE Trans Sig Proc 1999.
      http://classes.engr.oregonstate.edu/eecs/winter2009/ece464/AnalyticSignal_Sept1999_SPTrans.pdf
    """
    if pad_factor is not None and pad_factor >= 1:
      signal, padding = pad_signal(signal, pad_factor)

    n = signal.shape[0]
    assert np.mod(n, 2) == 0  # likely overly stringent but just for safety here

    nr = int(np.floor(n / 2)) + 1
    n_filts = filters.shape[0]

    # note: the real fft
    Fr = np.fft.rfft(signal)  # len: nr = n/2+1 (e.g., 8001 for 1 sec and 16K sr)

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
      analytic_subbands = analytic_subbands[:, :signal.shape[0]-padding]  # i dont know if this is correct
      subband_envelopes = subband_envelopes[:, :signal.shape[0]-padding]  # i dont know if this is correct

    subband_envelopes = apply_envelope_downsample(subband_envelopes, downsample)
    subband_envelopes = apply_envelope_nonlinearity(subband_envelopes, nonlinearity)

    # ## now just return things
    # if get_analytic_subbands_too and do_return_envs:
    #   return subband_envelopes, analytic_subbands
    # elif get_analytic_subbands_too:
    #   return subbands, analytic_subbands
    # elif do_return_envs:
    #   return subband_envelopes
    # else:
    #     raise NotImplementedError()

    return subband_envelopes


def generate_subbands(signal, filters, pad_factor=None):
    """
    adapted from jhm's generate_subbands.m

    generate subbands for a given signal [np array]
    -- uses the filters specified with self.makeErbCosFilters()

    this is the original code; i am since using a much expedited version
      which is directly above
    """
    # note: numpy defaults to row vecs
    # if pad_factor is not None and pad_factor >= 1:
    #   padding = signal.shape[0] * pad_factor - signal.shape[0]
    #   print('padding ', padding)
    #   signal = np.concatenate((signal, np.zeros(padding)))
    signal, padding = pad_signal(signal, pad_factor)

    fft_sample = np.fft.fft(signal)

    # below: fft_subbands and then subbands in time
    #   python implicitly expands out fft_sample!  awesome.
    subbands = filters * fft_sample
    subbands = np.real(np.fft.ifft(subbands))  # operates row-wise
    print('sbsh ,', subbands.shape)
    print(signal.shape)

    if pad_factor is not None and pad_factor >= 1:
      print(padding)
      subbands = subbands[:, :signal.shape[0]-padding]  # i dont know if this is correct
    print('sbsh post, ', subbands.shape)

    return subbands

def generate_analytic_subbands(signal, filters, pad_factor=None):
  subbands = generate_subbands(signal, filters, pad_factor=pad_factor)
  # subbands = subbands[:, :signal.shape[0]/2]
  return scipy.signal.hilbert(subbands)


def generate_subband_envelopes(signal, filters, pad_factor=None, downsample=None, nonlinearity=None):
  analytic_subbands = generate_analytic_subbands(signal, filters, pad_factor=pad_factor)
  subband_envelopes = np.abs(analytic_subbands)
  print(subband_envelopes.min(), ', ', subband_envelopes.max())

  subband_envelopes = apply_envelope_downsample(subband_envelopes, downsample)
  subband_envelopes = apply_envelope_nonlinearity(subband_envelopes, nonlinearity)

  return subband_envelopes


def pad_signal(signal, pad_factor):
  if pad_factor is not None and pad_factor >= 1:
    padding = signal.shape[0] * pad_factor - signal.shape[0]
    print('padding signal with ', padding)
    return (np.concatenate((signal, np.zeros(padding))), padding)


def apply_envelope_downsample(subband_envelopes, downsample):
  # DOWNSAMPLING BEFORE NONLINEARITY
  # was BadCoefficients error with Chebyshev type I filter [default]
  #   resample uses a fourier method and is needlessly long...
  if downsample is None:
    pass
  elif callable(downsample):
    # apply the downsampling function
    subband_envelopes = downsample(subband_envelopes)
  else:
    # assume that downsample is the downsampling factor
    subband_envelopes = scipy.signal.decimate(subband_envelopes, downsample, axis=1, ftype='fir') # this caused weird banding artifacts
    # subband_envelopes = scipy.signal.resample(subband_envelopes, np.ceil(subband_envelopes.shape[1]*(6000/SR)), axis=1)  # fourier method: this causes NANs that get converted to 0s
    # subband_envelopes = scipy.signal.resample_poly(subband_envelopes, 6000, SR, axis=1)  # this requires v0.18 of scipy
  subband_envelopes[subband_envelopes < 0] = 0
  return subband_envelopes


def apply_envelope_nonlinearity(subband_envelopes, nonlinearity):
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


def make_cochleagram_ray():
  DUR = 50 / 1000
  SR = 20001
  LOW_LIM = 50
  HI_LIM = 20000
  N_HUMAN = np.floor(freq2erb(HI_LIM) - freq2erb(LOW_LIM)) - 1;
  N_HUMAN = N_HUMAN.astype(int)
  ENV_SR = 6000
  PAD_FACTOR = 1
  Q = SR // ENV_SR
  F0 = 100

  # t = utils.matlab_arange(0, 200/1000, SR)
  t = np.arange(0, DUR + 1/SR, 1/SR)
  ct = np.zeros_like(t)
  for i in range(1,40+1):
    ct += np.sin(2*np.pi*F0*i*t)
  # ct = np.hstack((ct, np.zeros_like(ct)))

  print(t.shape)

  filts, hz_cutoffs, freqs = make_erb_cos_filters_nx(len(ct), SR, N_HUMAN, LOW_LIM, HI_LIM,
                                                     2, pad_factor=PAD_FACTOR, strict=False)
  # filts, hz_cutoffs, freqs = make_erb_cos_filters(len(ct), SR, N_HUMAN, LOW_LIM, HI_LIM, full_filter=True, strict=False)

  downsample_fx = lambda x: scipy.signal.resample_poly(x, 6000, SR, axis=1)
  # sub_envs = generate_subband_envelopes(ct, filts, pad_factor=PAD_FACTOR, downsample=downsample_fx, nonlinearity='log')
  sub_envs = generate_subband_envelopes_fast(ct, filts, pad_factor=PAD_FACTOR, downsample=downsample_fx, nonlinearity='log')

  # img = np.flipud(sub_envs.T)
  img = np.flipud(sub_envs)
  print('sub env shape: ', sub_envs.shape )
  # plt.imshow(img, cmap='inferno')
  plt.matshow(img, cmap='inferno')
  plt.show()


def main():
  make_cochleagram_ray()


if __name__ == '__main__':
  main()
