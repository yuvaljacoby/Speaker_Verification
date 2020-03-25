import os
import librosa
import numpy as np
from tqdm import tqdm
import pickle

BASE_PATH = "/home/yuval/projects/Speaker_Verification"

DATA_PATH = os.path.join("/home/yuval/projects/", "data/speaker_identification")
CLUSTER_DATA_PATH = "/cs/labs/guykatz/yuvalja/data/"
if os.path.exists(CLUSTER_DATA_PATH):
    DATA_PATH = CLUSTER_DATA_PATH

RAW_DATA_PATH = os.path.join(BASE_PATH, "data/VCTK_NEW/wav48_silence_trimmed")

# utter_min_len = (180 * 0.01 +  0.025) * 8000
# for interval in intervals:
#     if (interval[1] - interval[0]) > utter_min_len:  # If partial utterance is sufficient long,
#         utter_part = arr[interval[0]:interval[1]]  # save first and last 180 frames of spectrogram.
#         S = librosa.core.stft(y=utter_part, n_fft=512,
#                               win_length=int(0.025 * 8000), hop_length=int(0.01 * 8000))
#         S = np.abs(S) ** 2
#         mel_basis = librosa.filters.mel(sr=8000, n_fft=512, n_mels=40)
#         S = np.log10(np.dot(mel_basis, S) + 1e-6)  # log mel spectrogram of utterances

def wav_to_np(utter_path, config):
    '''
    :param path: path to wav file
    '''
    utter_min_len = (config.tisv_frame * config.hop + config.window) * config.sr  # lower bound of utterance length
    utter, sr = librosa.core.load(utter_path, config.sr)  # load utterance audio
    intervals = librosa.effects.split(utter, top_db=20)  # voice activity detection
    S = None
    for interval in intervals:
        if (interval[1] - interval[0]) > utter_min_len:  # If partial utterance is sufficient long,
            utter_part = utter[interval[0]:interval[1]]  # save first and last 180 frames of spectrogram.
            S = librosa.core.stft(y=utter_part, n_fft=512,
                                  win_length=int(200), hop_length=int(80))

            S = np.abs(S) ** 2
            mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
            S = np.log10(np.dot(mel_basis, S) + 1e-6)  # log mel spectrogram of utterances
    return S

def get_data_per_user(speaker_path, config):
    '''
    :param speaker_path: path to directory with wav files of the same user
    :return: list of numpy arrays, order by file name
    '''
    utterances_spec = []
    paths = []
    for utter_name in sorted(os.listdir(speaker_path)):
        utter_path = os.path.join(speaker_path, utter_name)  # path of each utterance
        S = wav_to_np(utter_path, config)
        if S is not None:
            utterances_spec.append(S[:, :config.tisv_frame])  # first 180 frames of partial utterance
            utterances_spec.append(S[:, -config.tisv_frame:])  # last 180 frames of partial utterance
            paths += [utter_path, utter_path]

    utterances_spec = np.array(utterances_spec)
    assert len(paths) == utterances_spec.shape[0]
    return utterances_spec, paths


def load_all_data(path, config, train_percent=0.8, use_cache=False):
    '''
    :param path: to directory where there are directories with wav file per user
    :param config: config for data load (n_mel, nfft etc...)
    :return: X_train, X_test, y_train, y_test
    '''
    if use_cache:
        X_train = pickle.load(open(os.path.join(DATA_PATH, "X_train.pkl"), "rb"))
        X_test = pickle.load(open(os.path.join(DATA_PATH, "X_test.pkl"), "rb"))
        y_train = pickle.load(open(os.path.join(DATA_PATH, "y_train.pkl"), "rb"))
        y_test = pickle.load(open(os.path.join(DATA_PATH, "y_test.pkl"), "rb"))
        index_path_map = pickle.load(open(os.path.join(DATA_PATH, "index_path_map.pkl"), "rb"))
        label_to_speaker = pickle.load(open(os.path.join(DATA_PATH, "label_to_speaker.pkl"), "rb"))
        print("############### WARNING USING CACHE #######################")
        return X_train, X_test, y_train, y_test, index_path_map, label_to_speaker

    classes_path = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]#[:10]
    index_path_map = {'train': {}, 'test': {}}
    label_to_speaker = {}
    X_train, X_test = np.array([]), np.array([])
    for i, d in enumerate(tqdm(classes_path)):
        X, paths = get_data_per_user(os.path.join(path, d), config)
        cut_idx = int(len(X) * train_percent)
        label_to_speaker[i] = d
        for k in range(len(X)):
            if k < cut_idx:
                index_path_map['train'][k + X_train.shape[0]] = paths[k]
            else:
                index_path_map['test'][k - cut_idx + X_test.shape[0]] = paths[k]
        if i == 0:
            X_train = X[:cut_idx, :, :]
            X_test = X[cut_idx:, :, :]
            y_train = np.array([i] * cut_idx)
            y_test = np.array([i] * (len(X) - cut_idx))
        else:
            X_train = np.vstack((X_train, X[:cut_idx, :, :]))
            X_test = np.vstack((X_test, X[cut_idx:, :, :]))
            y_train = np.hstack((y_train, np.array([i] * cut_idx)))
            y_test = np.hstack((y_test, np.array([i] * (X.shape[0] - cut_idx))))
    assert len(index_path_map['train'].keys()) == X_train.shape[0]
    assert len(index_path_map['test'].keys()) == X_test.shape[0]
    return X_train, X_test, y_train, y_test, index_path_map, label_to_speaker


if __name__ == "__main__":
    from model import config

    X_train, X_test, y_train, y_test, index_path_map, label_to_speaker = load_all_data(RAW_DATA_PATH, config)
    os.makedirs(DATA_PATH, exist_ok=True)
    import pickle

    pickle.dump(X_train, open(os.path.join(DATA_PATH, "X_train.pkl"), "wb"))
    pickle.dump(X_test, open(os.path.join(DATA_PATH, "X_test.pkl"), "wb"))
    pickle.dump(y_train, open(os.path.join(DATA_PATH, "y_train.pkl"), "wb"))
    pickle.dump(y_test, open(os.path.join(DATA_PATH, "y_test.pkl"), "wb"))
    pickle.dump(index_path_map, open(os.path.join(DATA_PATH, "index_path_map.pkl"), "wb"))
    pickle.dump(label_to_speaker, open(os.path.join(DATA_PATH, "label_to_speaker.pkl"), "wb"))

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    # assert len(X_test) == len(y_test)
