#create spectrograms
import random
import numpy as np
from numpy.lib import stride_tricks
import os
import scipy.io.wavfile as wav
"""
This script creates spectrogram matrices from wav files that can be passed \
to the CNN. This was heavily adopted from Frank Zalkow's work.
"""


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    """
    Short-time Fourier transform of audio signal.
    """
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(samples, shape=(int(cols), int(frameSize)),
                                      strides=(samples.strides[0]*hopSize,
                                      samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


def logscale_spec(spec, sr=44100, factor=20.):
    """
    Scale frequency axis logarithmically.
    """
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs


def stft_matrix(audiopath, binsize=2**10, png_name='tmp.png',
                save_png=False, offset=0):
    """
    A function that converts a wav file into a spectrogram represented by a \
    matrix where rows represent frequency bins, columns represent time, and \
    the values of the matrix represent the decibel intensity. A matrix of \
    this form can be passed as input to the CNN after undergoing normalization.
    """
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)

    ims = np.transpose(ims)
    ims = np.flipud(ims)  # weird - not sure why it needs flipping

    if save_png:
        create_png(ims, png_name)

    return ims

def get_random_samples(matrix, n_samples, crop_width):
    """
    Get N random samples with width of crop_width from the numpy matrix
    representing the participant's audio spectrogram.
    """
    # crop full spectrogram into segments of width = crop_width
    clipped_mat = matrix[:, (matrix.shape[1] % crop_width):]
    n_splits = clipped_mat.shape[1] / crop_width
    cropped_sample_ls = np.split(clipped_mat, n_splits, axis=1)
    # get random samples
    samples = random.sample(cropped_sample_ls, int(n_samples))

    return samples