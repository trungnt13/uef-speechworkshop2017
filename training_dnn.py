# ===========================================================================
# NOTE: the performance of float16 and float32 dataset are identical
# ===========================================================================
from __future__ import print_function, absolute_import, division

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import os
os.environ['ODIN'] = 'float32,gpu,tensorflow'
import sys

import numpy as np
np.random.seed(1208)

from odin import nnet as N, backend as K, fuel as F
from odin.basic import has_roles, BIAS, WEIGHT
from odin.stats import train_valid_test_split, freqcount
from odin import training

# ===========================================================================
# Get wav and process new dataset configuration
# ===========================================================================
# ====== process new features ====== #
data_path = os.path.join(os.path.dirname(sys.argv[0]), 'data', 'sample1.wav')
ds = F.load_digit_audio()
print(ds)
nb_classes = 10 # 10 digits (0-9)

# ===========================================================================
# Create feeder
# ===========================================================================
indices = [(name, start, end) for name, (start, end) in ds['indices']]
longest_utterances = max(int(end) - int(start) - 1
                         for i, start, end in indices)
longest_vad = max(end - start
                  for name, vad in ds['vadids'] for (start, end) in vad)
print("Longest Utterance:", longest_utterances)
print("Longest Vad:", longest_vad)


np.random.shuffle(indices)
train, valid, test = train_valid_test_split(indices, train=0.6, inc_test=True)
print('Nb train:', len(train), freqcount([int(i[0][0]) for i in train]))
print('Nb valid:', len(valid), freqcount([int(i[0][0]) for i in valid]))
print('Nb test:', len(test), freqcount([int(i[0][0]) for i in test]))

train_feeder = F.Feeder(ds['mspec'], train, ncpu=1)
test_feeder = F.Feeder(ds['mspec'], test, ncpu=2)
valid_feeder = F.Feeder(ds['mspec'], valid, ncpu=2)

recipes = [
    F.recipes.Name2Trans(converter_func=lambda x: int(x[0])),
    F.recipes.Normalization(
        mean=ds['mspec_mean'],
        std=ds['mspec_std'],
        local_normalize=False
    ),
    F.recipes.Sequencing(frame_length=longest_utterances, hop_length=1,
                         end='pad', endvalue=0, endmode='post',
                         transcription_transform=lambda x: x[-1]),
    F.recipes.CreateFile()
]

train_feeder.set_recipes(recipes)
test_feeder.set_recipes(recipes)
valid_feeder.set_recipes(recipes)
print('Feature shape:', train_feeder.shape)
feat_shape = (None,) + train_feeder.shape[1:]

X = K.placeholder(shape=feat_shape, name='X')
y = K.placeholder(shape=(None,), dtype='int32', name='y')

# ===========================================================================
# Create network
# ===========================================================================
f = N.Sequence([
    N.Dimshuffle(pattern=(0, 1, 2, 'x')),
    N.Conv(num_filters=32, filter_size=3, pad='same', strides=1,
           activation=K.linear),
    N.BatchNorm(activation=K.relu),
    N.Conv(num_filters=64, filter_size=3, pad='same', strides=1,
           activation=K.linear),
    N.BatchNorm(activation=K.relu),
    N.Pool(pool_size=2, strides=None, pad='valid', mode='max'),
    N.Flatten(outdim=3),

    # ====== RNN ====== #
    N.AutoRNN(128, rnn_mode='lstm', num_layers=3,
              direction_mode='bidirectional', prefer_cudnn=True),

    # ====== Dense ====== #
    N.Flatten(outdim=2),
    # N.Dropout(level=0.2), # adding dropout does not help
    N.Dense(num_units=1024, activation=K.relu),
    N.Dense(num_units=512, activation=K.relu),
    N.Dense(num_units=nb_classes, activation=K.softmax)
], debug=True)

K.set_training(True); y_train = f(X)
K.set_training(False); y_score = f(X)

# ====== create cost ====== #
cost_train = K.mean(K.categorical_crossentropy(y_train, y))
cost_test1 = K.mean(K.categorical_crossentropy(y_score, y))
cost_test2 = K.mean(K.categorical_accuracy(y_score, y))
cost_test3 = K.confusion_matrix(y_score, y, labels=range(10))

# ====== create optimizer ====== #
parameters = [p for p in f.parameters if has_roles(p, [WEIGHT, BIAS])]
optimizer = K.optimizers.RMSProp(lr=0.0001)
# ===========================================================================
# Standard trainer
# ===========================================================================
trainer, hist = training.standard_trainer(
    train_data=train_feeder, valid_data=valid_feeder, test_data=test_feeder,
    cost_train=cost_train, cost_score=[cost_test1, cost_test2], cost_regu=None,
    parameters=parameters, optimizer=optimizer,
    confusion_matrix=cost_test3, gradient_norm=True,
    batch_size=4, nb_epoch=3, valid_freq=0.8,
    save_path=None, save_obj=None,
    report_path='/tmp/tmp.pdf',
    enable_rollback=True, stop_callback=None, save_callback=None,
    labels=None
)
trainer.run()
