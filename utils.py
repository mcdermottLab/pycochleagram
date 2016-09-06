from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def matlab_arange(start, stop, num):
  """
    Mimics MATLAB's sequence generation.

  """
  return np.linspace(start, stop, num + 1)


