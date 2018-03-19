import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import scipy.io as sio
import numpy as np

import erbfilter as erb


import pdb
import matplotlib.pyplot as plt


def test_erb_filters(rfn, mode='all', verbose=0):
  # load matlab-generated reference file
  mlab = sio.loadmat(rfn)
  # print(list(mlab.keys()))

  # extract parameters used to generate these filterbanks
  signal_length = mlab['signal_length'][0, 0]  # everything is in an 2D array
  sr = mlab['sr'][0, 0]
  N = mlab['N'][0, 0]
  low_lim = mlab['low_lim'][0, 0]
  hi_lim = mlab['hi_lim'][0, 0]

  sample_factor_list = [1, 2, 4]
  matlab_api_fx_list = [erb.make_erb_cos_filters_1x, erb.make_erb_cos_filters_2x, erb.make_erb_cos_filters_4x]
  # use 1x, 2x, and 4x oversampling to test make_erb_cos_filters_nx
  try:
    for i in range(len(sample_factor_list)):
      sample_factor = sample_factor_list[i]

      # get keynames for looking up matlab reference values
      freqs_key = 'freqs_%s' % sample_factor
      hz_cutoffs_key = 'Hz_cutoffs_%s' % sample_factor
      filts_key = 'filts_%s' % sample_factor

      if verbose > 1:
        print('<Input Params: signal_length: %s, sr: %s, N: %s, low_lim: %s, hi_lim: %s, sample_factor: %s>' %
              (signal_length, sr, N, low_lim, hi_lim, sample_factor))

      if mode == 'all' or mode == 'nx':
        # test make_erb_cos_filters_nx
        filts, hz_cutoffs, freqs = erb.make_erb_cos_filters_nx(signal_length, sr, N,
                                                               low_lim, hi_lim, sample_factor,
                                                               pad_factor=None, full_filter=False, strict=False)
        assert np.allclose(filts, mlab[filts_key]), 'filts mismatch: make_erb_cos_filters_nx(...)'
        assert np.allclose(freqs, mlab[freqs_key]), 'Freqs mismatch: make_erb_cos_filters_nx(...)'
        assert np.allclose(hz_cutoffs, mlab[hz_cutoffs_key]), 'Hz_cutoff mismatch: make_erb_cos_filters_nx(...)'
        if verbose > 0:
          print('PASSED: make_erb_cos_filters_nx(%s)' % sample_factor)

      if mode == 'all' or mode == 'matlab':
        # test convenience function (i.e., the matlab api)
        matlab_api_fx = matlab_api_fx_list[i]
        filts, hz_cutoffs, freqs = matlab_api_fx(signal_length, sr, N, low_lim, hi_lim)
        assert np.allclose(filts, mlab[filts_key]), 'filts mismatch: %s(...)' % matlab_api_fx.__name__
        assert np.allclose(freqs, mlab[freqs_key]), 'Freqs mismatch: %s(...)' % matlab_api_fx.__name__
        assert np.allclose(hz_cutoffs, mlab[hz_cutoffs_key]), 'Hz_cutoff mismatch: %s(...)' % matlab_api_fx.__name__
        if verbose > 0:
          print('PASSED: %s' % matlab_api_fx.__name__)

      if mode == 'all' or mode == 'literal':
        # test the literal port of 1x
        if sample_factor == 1:
          filts, hz_cutoffs, freqs = erb.make_erb_cos_filters(signal_length, sr, N, low_lim, hi_lim)
          assert np.allclose(filts, mlab[filts_key]), 'filts mismatch: make_erb_cos_filters(...) (literal)'
          assert np.allclose(freqs, mlab[freqs_key]), 'Freqs mismatch: make_erb_cos_filters(...) (literal)'
          assert np.allclose(hz_cutoffs, mlab[hz_cutoffs_key]), 'Hz_cutoff mismatch: make_erb_cos_filters(...) (literal)'
          if verbose > 0:
            print('PASSED: make_erb_cos_filters (literal)')
  except AssertionError as e:
    print('FAILED')
    print('<Input Params: signal_length: %s, sr: %s, N: %s, low_lim: %s, hi_lim: %s, sample_factor: %s>' %
            (signal_length, sr, N, low_lim, hi_lim, sample_factor))
    print('filts: %s python: %s, (%s, %s), matlab: %s, (%s, %s)' % (np.abs(filts - mlab[filts_key]).max(), filts.shape, filts.min(), filts.max(),
                                                           mlab[filts_key].shape, mlab[filts_key].min(), mlab[filts_key].max()))
    print('freqs: %s python: %s, (%s, %s), matlab: %s, (%s, %s)' % (np.abs(freqs - mlab[freqs_key]).max(), freqs.shape, freqs.min(), freqs.max(),
                                                           mlab[freqs_key].shape, mlab[freqs_key].min(), mlab[freqs_key].max()))
    print('hz_cutoffs: %s python: %s, (%s, %s), matlab: %s, (%s, %s)' % (np.abs(hz_cutoffs - mlab[hz_cutoffs_key]).max(), hz_cutoffs.shape, hz_cutoffs.min(), hz_cutoffs.max(),
                                                           mlab[hz_cutoffs_key].shape, mlab[hz_cutoffs_key].min(), mlab[hz_cutoffs_key].max()))
    pdb.set_trace()
    raise(e)

def test_run_dir(in_path, verbose=0):
  fntp = [os.path.join(in_path, f) for f in os.listdir(in_path) if 'erb_human_filters_test' in f and f.endswith('.mat')]
  for i, f in enumerate(fntp):
    test_erb_filters(f, verbose=verbose)
    print('passed %s tests' % (i + 1))


def main():
  DIR = '/Users/raygon/Desktop/mdLab/projects/cochleagram/test/data/output/erb_human'
  # test_erb_filters()
  test_run_dir(DIR, verbose=1)


if __name__ == '__main__':
  main()
