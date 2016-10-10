from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def matlab_arange(start, stop, num):
  """Mimics MATLAB's sequence generation.

  Returns `num + 1` evenly spaced samples, calculated over the interval
  [`start`, `stop`].

  Args:
    start (scalar): The starting value of the sequence.
    stop (scalar): The end value of the sequence.
    num (int): Number of samples to generate.

  Returns:
    samples (ndarray): There are `num + 1` equally spaced samples in the closed
      interval.
  """
  return np.linspace(start, stop, num + 1)
