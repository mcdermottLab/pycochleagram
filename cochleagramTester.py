import os
import audioUtils


def main():
  IN_PATH = '/Users/raygon/Desktop/mdLab/sounds/naturalsounds165'
  CHUNKSIZE = 1024
  fntp = [os.path.join(IN_PATH, f) for f in os.listdir(IN_PATH) if not f.startswith('.')]
  fntp = fntp[4:5]
  print fntp

  for rfn in fntp:
    # load and preprocess sound
    snd, sampFreq = audioUtils.wav_to_array(rfn, rescale='standardize')

    # audio playback
    pyaudio_params={'channels': audioUtils.get_channels(snd),
                    'rate': sampFreq,
                    'frames_per_buffer': CHUNKSIZE,
                    'output': True,
                    'output_device_index': 1}
    audioUtils.play_array(snd, pyaudio_params=pyaudio_params)


if __name__ == '__main__':
  main()
