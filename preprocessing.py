import torch
import wavio
import librosa

N_FFT = 512

audio_kwargs = dict(
            sr=16000,
            N_FFT=N_FFT,
            hop_length=160,
            frame_length=256,
            n_mels=64,
            top_db=20
        )

def get_spectrogram_feature(filepath, **kwargs):
    (rate, width, sig) = wavio.readwav(filepath)
    sig = sig.ravel()

    # (N, T, 2) => N: number of frequencies, T: total number of frames used 
    stft = torch.stft(torch.FloatTensor(sig),
                        kwargs['N_FFT'],
                        hop_length=int(0.01*kwargs['sr']),
                        win_length=int(0.030*kwargs['sr']),
                        window=torch.hamming_window(int(0.030*kwargs['sr'])),
                        center=False,
                        normalized=False,
                        onesided=True)

    stft = (stft[:,:,0].pow(2) + stft[:,:,1].pow(2)).pow(0.5);
    amag = stft.numpy();
    feat = torch.FloatTensor(amag)
    feat = torch.FloatTensor(feat).transpose(0, 1)
    # (T, N)
    return feat



def compute_melspec(wav, hop_length, n_fft, n_mels, sr):
    melspec = librosa.feature.melspectrogram(wav,
                                             sr=sr,
                                             n_fft=n_fft,
                                             hop_length=hop_length,
                                             n_mels=n_mels)
    logmel = librosa.core.power_to_db(melspec)
    logmel = mel_0_1(logmel)
    return logmel

def mel_0_1(x):
    min_val = -81.38
    max_val = 25.57
    return (x - min_val) / (max_val - min_val)

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
    feat = torch.FloatTensor(logmel)
    return feat