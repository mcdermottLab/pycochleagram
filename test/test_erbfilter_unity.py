import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np

import erbfilter as erb


def test_erb_filts_unity(mode='grid', verbose=0):
  """Test that the squard ERB filterbank sums to 1.

    This is intended to check the generalization of 1x, 2x and 4x filter
    oversampling to nx.

    Args:
      mode ({'grid', 'rand'}): Determine how the sample_factor list will be
        tested. If 'grid' (default), the entire range from [1, 20] will be
        searched. If 'rand', random samples will be taken from a range.
      verbose (int): Controls output verbosity level.
  """
  mode = mode.lower()
  if mode == 'grid':
      sample_factor_list = range(1, 20)
  elif mode == 'rand':
    sample_factor_list = np.random.randint(1, 500, 20)
  else:
    raise NotImplementedError()

  signal_length_range = (1, 10000, 10)
  sr_range = (1000, 64000, 10)
  pad_factor = None
  low_lim_range = (1, 200, 10)
  hi_lim_range = (201, 60000, 10)
  N_range = (1, 1000, 10)
  ctr = 0

  for signal_length in np.random.randint(*signal_length_range):
    for sr in np.random.randint(*sr_range):
      for low_lim in np.random.randint(*low_lim_range):
        for hi_lim in np.random.randint(*hi_lim_range):
          for N in np.random.randint(*N_range):
            for sample_factor in sample_factor_list:
              try:
                if verbose > 0:
                  print('N: %s, sample_factor: %s, signal_length: %s, sr: %s, low_lim: %s, hi_lim: %s, pad_factor: %s' %
                        (N, sample_factor, signal_length, sr, low_lim, hi_lim, pad_factor))
                filts, hz_cutoffs, freqs = erb.make_erb_cos_filters_nx(signal_length, sr, N,
                                                                       low_lim, hi_lim, sample_factor,
                                                                       pad_factor=pad_factor, full_filter=False, strict=False)
                # get filters into columns
                filts = filts.T

                if verbose > 1:
                  print('filts shape: %s, sample_factor: %s' % (filts.shape, sample_factor))

                filts_sum = np.sum(filts * filts, axis=0)

                if verbose > 1:
                  print('filts_sum (min, max): (%s, %s)' % (filts_sum.min(), filts_sum.max()))

                is_close_to_one = np.allclose(filts_sum, np.ones_like(filts_sum))
                assert(is_close_to_one)

                ctr += 1
                if verbose > 0:
                  print('PASSED: test %s' % ctr)

              except AssertionError as e:
                import matplotlib.pyplot as plt
                import pdb
                print('\nFAILED\n------')
                pdb.set_trace()


def main():
  # test all sample factors from [1, 20]
  test_erb_filts_unity(mode='grid', verbose=1)
  # test 20  sample factors randomly chosen from [1, 500]
  test_erb_filts_unity(mode='rand', verbose=1)


if __name__ == '__main__':
  main()
