import sys
import os
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from spekaer_identification.utils import load_all_data
CHECKPOINT_PERIOD = 50

MODEL_SAVE_DIR = 'VCTK_NEW_MODELS/AUTOMATIC_TRAIN'
from spekaer_identification.create_sbatch import MODELS_FOLDER
if os.path.exists(MODELS_FOLDER):
    MODEL_SAVE_DIR = MODELS_FOLDER

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # import tensorflow as tf
    import tensorflow.compat.v1 as tf
    from tensorflow.keras import layers
    from tensorflow.keras.utils import to_categorical

    tf.disable_v2_behavior()
RNN_NAME = 'rnn'
LSTM_NAME = 'lstm'
FC_NAME = 'fc'


class config:
    batch_size = 32  # Originaly they used N and M for number of spekares and utterance per speaker
    hidden = 32
    proj = 64
    num_layer = 1
    loss = None
    nfft = 512
    n_mels = 40
    hop = 0.01
    window = 0.025
    sr = 8000
    tisv_frame = 180
    # tdsv_frame = 80


def get_model_architecture2(n_classes, layers_config):
    model = tf.keras.Sequential()

    rnn_idx = [i for i in range(len(layers_config)) if
               layers_config[i][0] == RNN_NAME or layers_config[i][0] == LSTM_NAME]
    input_shape = config.n_mels

    for i, (name, dim) in enumerate(layers_config):
        if name == RNN_NAME or name == LSTM_NAME:
            # If last rnn layer
            if i == rnn_idx[-1]:
                return_sequence = False
            else:
                return_sequence = True
            if i == 0:
                if name == RNN_NAME:
                    model.add(layers.SimpleRNN(dim, input_shape=(None, input_shape), activation='relu',
                                               return_sequences=return_sequence))
                else:
                    model.add(layers.LSTM(dim, input_shape=(None, input_shape), return_sequences=return_sequence,
                                          activation='relu', recurrent_activation='relu'))
            else:
                if name == RNN_NAME:
                    model.add(layers.SimpleRNN(dim, activation='relu', return_sequences=return_sequence))
                else:
                    model.add(layers.LSTM(dim, return_sequences=return_sequence))
        elif name == FC_NAME:
            if i == 0:
                model.add(layers.Dense(dim, activation='relu', input_dim=input_shape))
            else:
                model.add(layers.Dense(dim, activation='relu'))

    model.add(layers.Dense(n_classes, activation='softmax'))
    return model
    pass


def get_model_architecture(n_classes, rnn_layers, rnn_dim, fc_layers, fc_dim):
    model = tf.keras.Sequential()

    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 32)
    # model.add(layers.SimpleRNN(32, input_shape=(config.n_mels, config.tisv_frame), return_sequences=True,  activation='relu')) #, activation='relu'
    for i in range(rnn_layers):
        if i == rnn_layers - 1:
            return_sequence = False
        else:
            return_sequence = True

        # if lstm:
        #     model.add(
        #         layers.LSTM(rnn_dim, input_shape=(None, config.tisv_frame), return_sequences=return_sequence))
        # else:
        model.add(layers.SimpleRNN(rnn_dim, input_shape=(None, config.n_mels), activation='relu',
                                   return_sequences=return_sequence))
    for i in range(fc_layers):
        model.add(layers.Dense(fc_dim, activation='relu'))

    # model.add(layers.Dense(len(np.unique(y_train)), activation='softmax'))
    model.add(layers.Dense(n_classes, activation='softmax'))
    return model


def train(model, X_train, X_test, y_train, y_test, index_path_map, index_to_label, epochs, model_name=None):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    y_train = to_categorical(np.array(y_train))
    y_test = to_categorical(np.array(y_test))
    if model_name:
        cp_mn = '_'.join(model_name.split('_')[:-1])
        cp_mn += '_{epoch:04d}.ckpt'
        cp_path = os.path.join(MODEL_SAVE_DIR, cp_mn)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(cp_path, verbose=1, period=CHECKPOINT_PERIOD)
        model.fit(X_train, y_train, epochs=epochs, verbose=1, use_multiprocessing=True, callbacks = [cp_callback])
        model.save(os.path.join(MODEL_SAVE_DIR, model_name))
    else:
        model.fit(X_train, y_train, epochs=epochs, verbose=1, use_multiprocessing=True)

    # test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    scores = model.predict(X_test)
    y_hat = np.argmax(scores, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    test_acc = sum(y_hat == true_labels) / len(y_hat)

    return test_acc


def load_data_and_train(model, path, epochs=50):
    X_train, X_test, y_train, y_test, index_path_map, index_to_label = load_all_data(path, config, use_cache=True)
    return train(model, X_train, X_test, y_train, y_test, index_path_map, index_to_label, epochs)


def create_configs(rnn_options: List[List[int]], fc_options: List[int], n_classes=20):
    configs = []
    for ro in rnn_options:
        for fo in fc_options:
            l_config = [('rnn', r) for r in ro]
            l_config += [('fc', 32)] * fo
            configs.append({'layers_config': l_config, 'n_classes': n_classes})

    return configs


def train_on_configs(configs: List[Dict], epochs_options: List):
    for model_config in configs:
        for epoch in epochs_options:
            model = get_model_architecture2(**model_config)
            train_idx = np.argwhere(y_train < model_config['n_classes'])
            test_idx = np.argwhere(y_test < model_config['n_classes'])
            cur_y_train = y_train[train_idx][:, 0]
            cur_X_train = X_train[train_idx][:, 0, :, :]
            cur_X_test = X_test[test_idx][:, 0, :, :]
            cur_y_test = y_test[test_idx][:, 0]
            layers_name = "_".join([str(l[0]) + str(l[1]) for l in model_config['layers_config']])
            model_name = "model_{}classes_{}_epochs{}.h5".format(model_config['n_classes'], layers_name, epoch)
            # model_name = None  # don't save models
            test_acc = train(model, cur_X_train, cur_X_test, cur_y_train, cur_y_test, index_path_map, index_to_label,
                             epoch, model_name)
            print("\n")
            print(model_config)
            print("test_acc: {}".format(test_acc))
            results.append({'epochs': epoch, 'test_acc': test_acc, **model_config})
            pbar.update(1)

if __name__ == "__main__":

    path = ''  # "/home/yuval/projects/Speaker_Verification/data/VCTK/wav48"
    X_train, X_test, y_train, y_test, index_path_map, index_to_label = load_all_data(path, config, use_cache=True)
    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)
    all_configs = [

        # {'layers_config': [("lstm", 2)], 'n_classes': 20},
        # {},
        {'layers_config': [("rnn", 12), ("fc", 32), ("fc", 32), ("fc", 32), ("fc", 32), ("fc", 32)], 'n_classes': 20},
        {'layers_config': [("rnn", 10), ("fc", 32), ("fc", 32), ("fc", 32), ("fc", 32), ("fc", 32)], 'n_classes': 20},
        {'layers_config': [("rnn", 8), ("fc", 32), ("fc", 32), ("fc", 32), ("fc", 32), ("fc", 32)], 'n_classes': 20},
        {'layers_config': [("rnn", 8), ("rnn", 6), ("fc", 32), ("fc", 32), ("fc", 32), ("fc", 32), ("fc", 32)],
         'n_classes': 20},
        {'layers_config': [("rnn", 6), ("rnn", 6), ("fc", 32), ("fc", 32), ("fc", 32), ("fc", 32), ("fc", 32)],
         'n_classes': 20},
        {'layers_config': [("rnn", 2), ("fc", 32), ("fc", 32), ("fc", 32), ("fc", 32), ("fc", 32)], 'n_classes': 20},
        {'layers_config': [("rnn", 4), ("fc", 32), ("fc", 32), ("fc", 32), ("fc", 32), ("fc", 32)], 'n_classes': 20},
    ]
    rnn_options = [[4,4], [4,4,4], [4,4,4,4], [8], [8,4], [8,8], [8,4,4], [12], [12,12], [16]]
    fc_options = [2,3,4]
    all_configs = create_configs(rnn_options, fc_options)
    results = []
    epochs_options = [200]

    if len(sys.argv) > 1:
        all_configs = [all_configs[int(sys.argv[1])]]

    pbar = tqdm(total=len(all_configs) * len(epochs_options))
    train_on_configs(all_configs, epochs_options)

    df = pd.DataFrame(results)
    # df.to_pickle('results.pkl')
    print("\n")
    print(df)
