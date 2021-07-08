import soundfile as sf
import numpy as np
import tensorflow.lite as tflite
import time
import config as cfg
import soundfile
import librosa
import pandas as pd
from dcase_util.data import ProbabilityEncoder
from dcase_util.data import DecisionEncoder
import scipy
from utilities.ManyHotEncoder import ManyHotEncoder

def read_audio(path, target_fs=None, **kwargs):
    """ Read a wav file
    Args:
        path: str, path of the audio file
        target_fs: int, (Default value = None) sampling rate of the returned audio file, if not specified, the sampling
            rate of the audio file is taken

    Returns:
        tuple
        (numpy.array, sampling rate), array containing the audio at the sampling rate given

    """
    (audio, fs) = soundfile.read(path, **kwargs)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def load_and_compute_mel_spec(wav_path):
    (audio, _) = read_audio(wav_path, sample_rate)
    if audio.shape[0] == 0:
        raise IOError("File {wav_path} is corrupted!")
    else:
        t1 = time.time()
        mel_spec = calculate_mel_spec(audio, True)
    return mel_spec

def calculate_mel_spec( audio, compute_log=False):
    """
    Calculate a mal spectrogram from raw audio waveform
    Note: The parameters of the spectrograms are in the config.py file.
    Args:
        audio : numpy.array, raw waveform to compute the spectrogram
        compute_log: bool, whether to get the output in dB (log scale) or not

    Returns:
        numpy.array
        containing the mel spectrogram
    """
    # Compute spectrogram
    ham_win = np.hamming(n_window)

    spec = librosa.stft(
        audio,
        n_fft= n_window,
        hop_length=hop_size,
        window=ham_win,
        center=True,
        pad_mode='reflect'
    )

    mel_spec = librosa.feature.melspectrogram(
        S=np.abs(spec),  # amplitude, for energy: spec**2 but don't forget to change amplitude_to_db.
        sr=sample_rate,
        n_mels=n_mels,
        fmin=mel_min_max_freq[0], fmax=mel_min_max_freq[1],
        htk=False, norm=None)
    if compute_log:
        mel_spec = librosa.amplitude_to_db(mel_spec)  # 10 * log10(S**2 / ref), ref default is 1
        mel_spec = mel_spec.T
        mel_spec = mel_spec.astype(np.float32)
        return mel_spec

mean_ = np.load('weights/mean.npy')
std_ = np.load('weights/std.npy')

sample_rate = cfg.sample_rate
n_window = cfg.n_window
hop_size = cfg.hop_size
n_mels = cfg.n_mels
mel_min_max_freq = (cfg.mel_f_min, cfg.mel_f_max)
threshold = 0.5
pooling_time_ratio  = 4

many_hot_encoder = ManyHotEncoder(labels=cfg.classes, n_frames=cfg.max_frames // pooling_time_ratio, )
interpreter_1 = tflite.Interpreter(model_path='./weights/model_final_lite.tflite')
interpreter_1.allocate_tensors()

input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()

states_1 = np.zeros(input_details_1[1]['shape']).astype('float32')

wav='E:/intern/project2/learn/dataset/audio/validation/validation/Y_WQV7YjYrX0_470.000_480.000.wav'
s,f=read_audio(wav)
mel_spec = load_and_compute_mel_spec(wav)
mel_spec = (mel_spec - mean_)/std_
mel_spec_shaped = np.reshape(mel_spec,(-1,628,128))
mel_spec_shaped = np.float32(mel_spec_shaped)
num_blocks = mel_spec_shaped.shape[0]

strong_pred=[]
weak_pred = []

for idx in range(num_blocks):
    inp = np.expand_dims(mel_spec_shaped[idx],axis = (0,3))
    interpreter_1.set_tensor(input_details_1[0]['index'], inp)
    interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
    interpreter_1.invoke()
    weak = interpreter_1.get_tensor(output_details_1[0]['index']) 
    strong = interpreter_1.get_tensor(output_details_1[1]['index']) 
    states_1 = interpreter_1.get_tensor(output_details_1[2]['index']) 
    strong_pred.append(strong)
    weak_pred.append(weak)

strong_pred=np.array(strong_pred)
weak_pred = np.array(weak_pred)

strong_pred = np.expand_dims(np.squeeze(strong_pred),axis=0)
out_nb_frames_1s = cfg.sample_rate / cfg.hop_size / pooling_time_ratio
median_window = max(int(cfg.median_window_s * out_nb_frames_1s), 1)

pred_strong_bin = ProbabilityEncoder().binarization(strong_pred,
                                                    binarization_type="global_threshold",
                                                    threshold=threshold)

pred_strong_m = scipy.ndimage.filters.median_filter(pred_strong_bin[0], (median_window, 1))

pred = many_hot_encoder.decode_strong(pred_strong_m)

pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
pred.loc[:, ["onset", "offset"]] *= pooling_time_ratio / (cfg.sample_rate / cfg.hop_size)
pred.loc[:, ["onset", "offset"]] = pred[["onset", "offset"]].clip(0, cfg.max_len_seconds)

print(pred)
pred.to_csv('outputs/lite_prediction.csv')