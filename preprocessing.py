import os
import torch
import wavio
import pickle 
import argparse
import librosa
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import nsml
from nsml import GPU_NUM, DATASET_PATH, DATASET_NAME, HAS_DATASET
import warnings
warnings.filterwarnings("ignore")


if HAS_DATASET == False:
    DATASET_PATH = './sample_dataset'

DATASET_PATH = os.path.join(DATASET_PATH, 'train')


N_FFT = 512
SAMPLE_RATE = 16000

audio_kwargs = dict(
            sr=16000,
            N_FFT=N_FFT,
            hop_length=160,
            frame_length=256,
            n_mels=64,
            top_db=20
        )

def get_spectrogram_feature(filepath):
    # (rate, width, sig) = wavio.readwav(filepath)
    # sig = sig.ravel()
    sig = librosa.load(filepath, SAMPLE_RATE)[0]
    sig = librosa.effects.trim(sig, frame_length=256, hop_length=160, top_db=20)[0]
    
    stft = torch.stft(torch.FloatTensor(sig),
                        N_FFT,
                        hop_length=int(0.01*SAMPLE_RATE),
                        win_length=int(0.030*SAMPLE_RATE),
                        window=torch.hamming_window(int(0.030*SAMPLE_RATE)),
                        center=False,
                        normalized=False,
                        onesided=True)

    stft = (stft[:,:,0].pow(2) + stft[:,:,1].pow(2)).pow(0.5);
    amag = stft.numpy();
    feat = torch.FloatTensor(amag)
    feat = torch.FloatTensor(feat).transpose(0, 1)
    feat = stft_0_1(feat)
    return feat

def stft_0_1(x):
    min_val = 0.0
    max_val = 69.57
    return (x - min_val) / (max_val - min_val)


def compute_melspec(wav, hop_length, n_fft, n_mels, sr):
    melspec = librosa.feature.melspectrogram(wav,
                                             sr=sr,
                                             n_fft=n_fft,
                                             hop_length=hop_length,
                                             n_mels=n_mels)
    logmel = librosa.core.power_to_db(melspec)
    logmel = mel_0_1(logmel)
    return logmel

def get_mel_features(filepath, **kwargs):
    wav = librosa.load(filepath, kwargs['sr'])[0]
    wav = librosa.effects.trim(wav,
                               frame_length = kwargs['frame_length'],
                               hop_length = kwargs['hop_length'],
                               top_db = kwargs['top_db'])[0]
    logmel = compute_melspec(wav,
                             hop_length = kwargs['hop_length'],
                             n_fft = kwargs['N_FFT'],
                             n_mels = kwargs['n_mels'],
                             sr = kwargs['sr'])
    logmel = torch.FloatTensor(logmel)
    feat = torch.FloatTensor(logmel).transpose(0, 1)
    return feat

def mel_0_1(x):
    min_val = -81.38
    max_val = 25.57
    return (x - min_val) / (max_val - min_val)



def bind_nsml():
    def pickle_save(dir_name):
        os.makedirs(dir_name, exist_ok=True)    
        with open(os.path.join(dir_name, 'audio_features.pkl'), 'wb') as handle:
            pickle.dump(audio_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("pickle saved")

    def pickle_load(dir_name):
        with open(os.path.join(dir_name, 'audio_features.pkl'), 'rb') as handle:
            audio_features = pickle.load(handle)
        print("pickle loaded")

    def infer(root, phase):
        return _infer(root, phase, model=model)

    nsml.bind(save=pickle_save, load=pickle_load, infer=infer)


def main():

    global audio_features
    
    parser = argparse.ArgumentParser(description='Speech hackathon Baseline')
    arg = parser.add_argument
    arg('--feature_type', type=str, default='stft', help='choose which feature to preprocess') # SGD
    arg('--num_cores', type=int, default=6, help='number of cores') # SGD
    args = parser.parse_args()


    bind_nsml()

    data_list = os.path.join(DATASET_PATH, 'train_data', 'data_list.csv')
    wav_paths = list()
    # script_paths = list()

    with open(data_list, 'r') as f:
        for line in f:
            # line: "aaa.wav,aaa.label"
            wav_path, script_path = line.strip().split(',')
            wav_paths.append(os.path.join(DATASET_PATH, 'train_data', wav_path))
            # script_paths.append(os.path.join(DATASET_PATH, 'train_data', script_path))

    num_cores = args.num_cores

    if args.feature_type == 'stft':
    
        audio_features = Parallel(n_jobs=num_cores)(
            delayed(lambda x: get_spectrogram_feature(path))(path) for path in tqdm(np.asarray(wav_paths)))

        save_name = 'stft_preprocessed'

    elif args.feature_type == 'mel':

        audio_features = Parallel(n_jobs=num_cores)(
            delayed(lambda x: get_mel_features(path, **audio_kwargs))(path) for path in tqdm(np.asarray(wav_paths)))
        
        save_name = 'mel_preprocessed'

    # save on nsml
    nsml.save(save_name)

if __name__ == "__main__":
    main()
