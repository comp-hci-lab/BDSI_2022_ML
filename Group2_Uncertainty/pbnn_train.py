import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_probability as tfp

print(tf.__version__)
print('Listing all GPU resources:')
print(tf.config.experimental.list_physical_devices('GPU'))
print()
import tensorflow.keras as keras
print(tfp.__version__)
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import pickle
import os
import sys
import git
import importlib.util

LAYER_NAME = os.getenv('LAYER_NAME')

FILTERS = 32
DATA_SIZE = 60000*144*144
PRIOR_MU = 0
PRIOR_SIGMA = 10

BATCH_SIZE = 256
EPOCHS = 400
VERBOSE = 2

ROOT_PATH = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
DATA_PATH = ROOT_PATH + "/data/"
LAYER_PATH = ROOT_PATH + "/layers/" + LAYER_NAME + "/"
SAVE_PATH = LAYER_PATH + LAYER_NAME + "_bayesian_model.h5"
PICKLE_PATH = LAYER_PATH + LAYER_NAME + '_hist.pkl'
MODEL_PATH = LAYER_PATH + LAYER_NAME + "_model"

print("-" * 30)
print("Constructing model...")
print("-" * 30)
mirrored_strategy = tf.distribute.MirroredStrategy()

spec = importlib.util.spec_from_file_location(MODEL_PATH, MODEL_PATH + ".py")
ModelLoader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ModelLoader)

with mirrored_strategy.scope():
    model = ModelLoader.make_model()

print('Model summary:')
print(model.summary())

print('Model losses:')
print(model.losses)

print("-" * 30)
print("Loading training and testing data...")
print("-" * 30)

Xy_train = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(np.load(DATA_PATH + 'imgs_train.npy')),
                                tf.data.Dataset.from_tensor_slices(np.load(DATA_PATH + 'msks_train.npy')))).cache().batch(BATCH_SIZE).prefetch(2)
Xy_test = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(np.load(DATA_PATH + 'imgs_test.npy')),
                                tf.data.Dataset.from_tensor_slices(np.load(DATA_PATH + 'msks_test.npy')))).cache().batch(BATCH_SIZE).prefetch(2)


print("-" * 30)
print("Fitting model with training data ...")
print("-" * 30)

print("Training the model started at {}".format(datetime.datetime.now()))
start_time = time.time()

#Train the model
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=SAVE_PATH,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history = model.fit(Xy_train,
          validation_data=Xy_test,
          epochs=EPOCHS, verbose=VERBOSE,
          callbacks=[model_checkpoint_callback])

print("Total time elapsed for training = {} seconds".format(time.time() - start_time))
print("Training finished at {}".format(datetime.datetime.now()))


# Save the model
# serialize weights to HDF5
model.save_weights(SAVE_PATH)
print("Saved model to disk (.h5)")

#Save training history
history_dict = history.history
with open(PICKLE_PATH, 'wb') as file_pi:
        pickle.dump(history_dict, file_pi)
print('Pickled training history')
