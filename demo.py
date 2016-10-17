from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

import utils
import cochleagram as cgram


def demo_human_cochleagram():
  """Demo to generate the human cochleagram of a tone synthesized with 40
  harmonics.
  """
  coch, sig_dict = cgram.demo_human_cochleagram(downsample=None, nonlinearity=None, interact=False)
  coch_log, sig_dict = cgram.demo_human_cochleagram(downsample=None, nonlinearity='log', interact=False)
  coch_pow, sig_dict = cgram.demo_human_cochleagram(downsample=None, nonlinearity='power', interact=False)

  plt.subplot(321)
  plt.title('Signal waveform')
  plt.plot(sig_dict['signal'])
  plt.ylabel('amplitude')
  plt.xlabel('time')

  plt.subplot(323)
  plt.title('Signal Frequency Content')
  f, Pxx_den = welch(sig_dict['signal'].flatten(), sig_dict['sr'], nperseg=1024)
  plt.semilogy(f, Pxx_den)
  plt.xlabel('frequency [Hz]')
  plt.ylabel('PSD [V**2/Hz]')

  plt.subplot(322)
  plt.title('Cochleagram with no nonlinearity')
  plt.ylabel('filter #')
  plt.xlabel('time')
  utils.cochshow(np.flipud(coch), interact=False)
  plt.gca().invert_yaxis()

  plt.subplot(324)
  plt.title('Cochleagram with nonlinearity: "log"')
  plt.ylabel('filter #')
  plt.xlabel('time')
  utils.cochshow(np.flipud(coch_log), interact=False)
  plt.gca().invert_yaxis()

  plt.subplot(326)
  plt.title('Cochleagram with nonlinearity: "power"')
  plt.ylabel('filter #')
  plt.xlabel('time')
  utils.cochshow(np.flipud(coch_pow), interact=False)
  plt.gca().invert_yaxis()
  plt.tight_layout()
  plt.show()


def demo_playback():
  IN_PATH = '/Users/raygon/Desktop/mdLab/sounds/naturalsounds165'
  CHUNKSIZE = 1024
  fntp = [os.path.join(IN_PATH, f) for f in os.listdir(IN_PATH) if not f.startswith('.')]
  fntp = fntp[4:5]
  print(fntp)

  for rfn in fntp:
    # load and preprocess sound
    snd, sampFreq = utils.wav_to_array(rfn, rescale='standardize')

    # audio playback
    pyaudio_params={'channels': utils.get_channels(snd),
                    'rate': sampFreq,
                    'frames_per_buffer': CHUNKSIZE,
                    'output': True,
                    'output_device_index': 1}
    utils.play_array(snd, rescale=None, pyaudio_params=pyaudio_params, ignore_warning=True)


def main():
  demo_human_cochleagram()
  # demo_playback()


if __name__ == '__main__':
  main()
