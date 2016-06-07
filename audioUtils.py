import numpy as np
from scipy.io import wavfile
import pyaudio
import matplotlib.pyplot as plt

def parse_rescale_arg(rescale):
  """ Parse the rescaling argument to a standard form. Throws an error if rescale
    value is unrecognized.
  """
  _rescale = rescale.lower()
  if _rescale == 'normalize':
    out_rescale = 'normalize'
  elif _rescale == 'standardize':
    out_rescale = 'standardize'
  elif _rescale is None or _rescale == '':
    out_rescale = None
  else:
    raise ValueError('Unrecognized rescale value: %s' % rescale)
  return out_rescale


def get_channels(snd_array):
  n_channels = 1
  if snd_array.ndim > 1:
    n_channels = snd_array.shape[1]
  return n_channels


def wav_to_array(fn, rescale='standardize'):
  """ Reads wav file data into a numpy array.

    Args:
      fn (str): path to .wav file
      normalize (str): Determines type of rescaling to perform. 'standardize' will
        divide by the max value allowed by the numerical precision of the input.
        'normalize' will rescale to the interval [-1, 1]. None or '' will not
        perform rescaling.

    Returns:
      snd, samp_freq (int, np.array): Sampling frequency of the input sound, followed
        by the sound as a numpy-array .
  """
  _rescale = parse_rescale_arg(rescale)
  samp_freq, snd = wavfile.read(fn)
  if _rescale == 'standardize':
    snd = snd / float(np.iinfo(snd.dtype).max)  # rescale so max value allowed by precision has value 1
  elif _rescale == 'normalize':
    snd = snd / float(snd.max())  # rescale to [-1, 1]
  # do nothing if rescale is None or ''
  return snd, samp_freq


def play_array(snd_array, pyaudio_params={}):
  _pyaudio_params = {'format': pyaudio.paFloat32,
                     'channels': 1,
                     'rate': 44100,
                     'frames_per_buffer': 1024,
                     'output': True,
                     'output_device_index': 1}

  for k, v in pyaudio_params.items():
    _pyaudio_params[k] = v

  print _pyaudio_params
  p = pyaudio.PyAudio()
  # stream = p.open(format=pyaudio.paFloat32,
  #                 channels=1,
  #                 rate=44100,
  #                 frames_per_buffer=1024,
  #                 output=True,
  #                 output_device_index=1)
  stream = p.open(**_pyaudio_params)
  data = snd_array.astype(np.float32).tostring()

  # stream = p.open(format=pyaudio.paInt16, channels=1, rate=samp_freq, output=True, frames_per_buffer=CHUNKSIZE)
  # data = snd.astype(snd.dtype).tostring()
  stream.write(data)


def plot_waveform(snd_array, samp_freq, n_channels=None):
  if n_channels is None:
    n_channels = get_channels(snd_array)

  print n_channels
  if snd_array.size % n_channels:  # if not 0
    raise ValueError('Odd amount of data in sound array')

  time_axis = np.arange(0, snd_array.size / n_channels, 1)
  time_axis = time_axis / float(samp_freq)
  time_axis = time_axis * 1000  # scale to milliseconds
  print time_axis[-1]
  plt.plot(time_axis, snd_array)
  return time_axis, snd_array
