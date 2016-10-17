import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import scipy.io as sio
import numpy as np

from utils import cochshow
import erbfilter as erb
import subband as sb
import cochleagram as cgram

import scipy.signal


import pdb
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'viridis'

import time


def test_subbands(rfn, erb_filter_mode='all', verbose=0):
  print('Testing subband equality for %s' % rfn)
  # load matlab-generated reference file
  mlab = sio.loadmat(rfn)
  # print(list(mlab.keys()))

  # extract parameters used to generate these filterbanks
  signal = mlab['wavData']  # everything is in an 2D array
  # remove 2x zero padding, if necessary
  if mlab['zp'][0, 0] == 1:
    signal = signal[:signal.shape[0]/2]
  print(np.shape(signal))

  signal_length = len(signal)
  sr = mlab['Fs'][0, 0]
  N = mlab['N_human'][0, 0]
  low_lim = mlab['low_lim_human'][0, 0]
  hi_lim = mlab['high_lim_human'][0, 0]

  sample_factor_list = [1, 2, 4]
  matlab_api_fx_list = [erb.make_erb_cos_filters_1x, erb.make_erb_cos_filters_2x, erb.make_erb_cos_filters_4x]
  # use 1x, 2x, and 4x oversampling to test make_erb_cos_filters_nx
  try:
    for i in range(len(sample_factor_list)):
      sample_factor = sample_factor_list[i]

      # get keynames for looking up matlab reference values
      freqs_key = 'freqs_%s' % sample_factor
      hz_cutoffs_key = 'Hz_cutoffs_%s' % sample_factor
      filts_key = 'fft_filts_%s' % sample_factor
      fft_sample_key = 'fft_sample_%s' % sample_factor
      subbands_key = 'subbands_%s' % sample_factor

      if verbose > 1:
        print('<Input Params: signal_length: %s, sr: %s, N: %s, low_lim: %s, hi_lim: %s, sample_factor: %s>' %
              (signal_length, sr, N, low_lim, hi_lim, sample_factor))

      if mode == 'all' or mode == 'nx':
        if verbose > 1:
          print('making erb filters')
        # test make_erb_cos_filters_nx
        filts, hz_cutoffs, freqs = erb.make_erb_cos_filters_nx(signal_length, sr, N,
                                                               low_lim, hi_lim, sample_factor,
                                                               pad_factor=1, full_filter=True, strict=False)
        if verbose > 1:
          print('performing subband decomposition')
        start_time = time.time()
        subbands_dict = sb.generate_subbands(signal.flatten(), filts, 1, debug_ret_all=True)
        tot_time = time.time() - start_time
        print('TIME --> %s' % tot_time)
        # print(subbands.shape, mlab[subbands_key].shape)
        # subbands = sb.generate_subbands(signal.flatten(), filts, 1, debug_ret_all=False)
        # print(list(subbands_dict.keys()))

        if verbose > 1:
          print('running assertions')
        # check *full* filterbanks and associated data
        assert np.allclose(filts, mlab[filts_key].T), 'filts mismatch: make_erb_cos_filters_nx(...)'  # transpose because of python
        assert np.allclose(freqs, mlab[freqs_key]), 'Freqs mismatch: make_erb_cos_filters_nx(...)'
        assert np.allclose(hz_cutoffs, mlab[hz_cutoffs_key]), 'Hz_cutoff mismatch: make_erb_cos_filters_nx(...)'
        if verbose > 0:
          print('PASSED ERB Filters: make_erb_cos_filters_nx(%s)' % sample_factor)

        # check variables associated with subband decomposition
        assert np.allclose(subbands_dict['fft_sample'], mlab['fft_sample_1'].flatten()), 'fft sample mismatch: make_erb_cos_filters_nx(...)'
        assert np.allclose(subbands_dict['subbands'], mlab[subbands_key].T), 'subband mismatch: make_erb_cos_filters_nx(...)'
        if verbose > 0:
          print('PASSED Subbands: make_erb_cos_filters_nx(%s)' % sample_factor)
        # pdb.set_trace()
      if erb_filter_mode == 'matlab':
        raise NotImplementedError()
      if erb_filter_mode == 'literal':
        raise NotImplementedError()
  except AssertionError as e:
    print('FAILED')
    print('<Input Params: signal_length: %s, sr: %s, N: %s, low_lim: %s, hi_lim: %s, sample_factor: %s>' %
            (signal_length, sr, N, low_lim, hi_lim, sample_factor))
    pdb.set_trace()
    raise(e)


def test_cochleagram(rfn, erb_filter_mode='all', coch_mode='fast',verbose=0):
  print('<Testing cochleagrams with coch_mode: %s>' % coch_mode)
  # load matlab-generated reference file
  mlab = sio.loadmat(rfn)
  # print(list(mlab.keys()))

  # get the function to generate the cochleagrams with
  coch_fx = _get_coch_function(coch_mode)

  # extract parameters used to generate these filterbanks
  signal = mlab['wavData']  # everything is in an 2D array
  # remove 2x zero padding, if necessary
  if mlab['zp'][0, 0] == 1:
    signal = signal[:signal.shape[0]/2]
  print(np.shape(signal))

  signal_length = len(signal)
  sr = mlab['Fs'][0, 0]
  N = mlab['N_human'][0, 0]
  low_lim = mlab['low_lim_human'][0, 0]
  hi_lim = mlab['high_lim_human'][0, 0]

  sample_factor_list = [1, 2, 4]
  # use 1x, 2x, and 4x oversampling to test make_erb_cos_filters_nx
  try:
    for i in range(len(sample_factor_list)):
      sample_factor = sample_factor_list[i]

      # get keynames for looking up matlab reference values
      freqs_key = 'freqs_%s' % sample_factor
      hz_cutoffs_key = 'Hz_cutoffs_%s' % sample_factor
      filts_key = 'fft_filts_%s' % sample_factor
      fft_sample_key = 'fft_sample_%s' % sample_factor

      if verbose > 1:
        print('<Input Params: signal_length: %s, sr: %s, N: %s, low_lim: %s, hi_lim: %s, sample_factor: %s>' %
              (signal_length, sr, N, low_lim, hi_lim, sample_factor))

      if erb_filter_mode == 'all' or erb_filter_mode == 'nx':
        if coch_mode == 'coch':
          sub_envs_key = 'sub_envs_%s' % sample_factor
          coch = cgram.human_cochleagram(signal, sr, N, low_lim, hi_lim, sample_factor, pad_factor=1, strict=False)
          assert np.allclose(coch, mlab[sub_envs_key].T), 'subband_env mismatch: make_erb_cos_filters_nx(...)' # transpose for matlab-to-python
          if verbose > 0:
            print('\tPASSED Quick Cochleagram: make_erb_cos_filters_nx(%s)' % sample_factor)
        else:
          # test make_erb_cos_filters_nx
          filts, hz_cutoffs, freqs = erb.make_erb_cos_filters_nx(signal_length, sr, N,
                                                                 low_lim, hi_lim, sample_factor,
                                                                 pad_factor=1, full_filter=True, strict=False)
          # pdb.set_trace()
          # check *full* filterbanks and associated data
          assert np.allclose(filts, mlab[filts_key].T), 'filts mismatch: make_erb_cos_filters_nx(...)'  # transpose because of python
          assert np.allclose(freqs, mlab[freqs_key]), 'Freqs mismatch: make_erb_cos_filters_nx(...)'
          assert np.allclose(hz_cutoffs, mlab[hz_cutoffs_key]), 'Hz_cutoff mismatch: make_erb_cos_filters_nx(...)'
          if verbose > 0:
            print('\tPASSED ERB Filters: make_erb_cos_filters_nx(%s)' % sample_factor)

        # pdb.set_trace()
        if coch_mode == 'subband':
          subbands_key = 'subbands_%s' % sample_factor

          # perform subband decomposition
          start_time = time.time()
          subbands_dict = sb.generate_subbands(signal.flatten(), filts, 1, debug_ret_all=True)
          tot_time = time.time() - start_time
          print('TIME --> %s' % tot_time)

          # check variables associated with subband decomposition
          # assert np.allclose(subbands_dict['fft_sample'], mlab[fft_sample_key].flatten()), 'fft sample mismatch: make_erb_cos_filters_nx(...)'
          assert np.allclose(subbands_dict['subbands'], mlab[subbands_key].T), 'subband mismatch: make_erb_cos_filters_nx(...)'
          if verbose > 0:
            print('\tPASSED Subbands: make_erb_cos_filters_nx(%s)' % sample_factor)
        elif coch_mode != 'coch':
          sub_envs_key = 'sub_envs_%s' % sample_factor

          # generate cochleagrams
          start_time = time.time()
          subband_envs = coch_fx(signal,filts, 1)
          tot_time = time.time() - start_time
          print('TIME --> %s' % tot_time)

          # check variables associated with subband envelopes (cochleagrams)
          assert np.allclose(subband_envs, mlab[sub_envs_key].T), 'subband_env mismatch: make_erb_cos_filters_nx(...)' # transpose for matlab-to-python
          if verbose > 0:
            print('\tPASSED Cochleagram: make_erb_cos_filters_nx(%s)' % sample_factor)
      if erb_filter_mode == 'matlab':
        raise NotImplementedError('Tests are only implemented for make_erb_cos_filters_nx')
      if erb_filter_mode == 'literal':
        raise NotImplementedError('Tests are only implemented for make_erb_cos_filters_nx')
  except AssertionError as e:
    print('\tFAILED')
    print('<Input Params: signal_length: %s, sr: %s, N: %s, low_lim: %s, hi_lim: %s, sample_factor: %s>' %
            (signal_length, sr, N, low_lim, hi_lim, sample_factor))
    pdb.set_trace()
    raise(e)


def _get_coch_function(mode):
  mode = mode.lower()
  if mode == 'fast':
    coch_fx = sb.generate_subband_envelopes_fast
  elif mode == 'alexfast':
    coch_fx = sb.generate_subband_envelopes_alex_fast
  elif mode == 'standard':
    coch_fx = sb.generate_subband_envelopes
  elif mode == 'subband' or mode == 'coch':
    coch_fx = None
  else:
    raise ValueError('Unrecognized coch_fx mode: %s' % mode)
  return coch_fx


def test_run_dir(in_path, verbose=0):
  fntp = [os.path.join(in_path, f) for f in os.listdir(in_path) if 'human_subands_test' in f and f.endswith('.mat')]
  for i, f in enumerate(fntp):
    if i < 0:
      print('skipped test %s' % (i + 1))
      continue

    test_cochleagram(f, erb_filter_mode='nx', coch_mode='fast', verbose=verbose)
    print('passed %s tests' % (i + 1))


def main():
  # DIR  ='/Users/Andrew/Projects/McDermott/py-cochleagram/py-cochleagram/test/data/cochleagram_outputs/'
  #DIR  ='/Users/raygon/Desktop/mdlab/projects/cochleagram/test/data/output/coch_human/'
  DIR = '/home/vagrant/Lab_Projects/py-cochleagram/test/data/output/coch_human'
  # test_erb_filters()
  test_run_dir(DIR, verbose=2)


if __name__ == '__main__':
  main()
