import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_probability as tfp
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
import importlib.util
import git

LAYER_NAME = os.getenv('LAYER_NAME')

FILTERS = 32
DATA_SIZE = 60000*144*144

BATCH_SIZE = 128
EPOCHS = 100
VERBOSE = 2

ROOT_PATH = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
DATA_PATH = "/scratch/precisionhealth_owned_root/precisionhealth_owned1/snehalbp/GBM_LGG_tradeoffs/ALL/" #Change datapath
LAYER_PATH = ROOT_PATH + "/models/" + LAYER_NAME + "/"
SAVE_PATH = LAYER_PATH + LAYER_NAME + "_model_weights.h5"
PICKLE_PATH = LAYER_PATH + LAYER_NAME + '_history.pkl'
MODEL_PATH = LAYER_PATH + LAYER_NAME + "_model"

print("-" * 30)
print("Constructing model...")
print("-" * 30)
mirrored_strategy = tf.distribute.MirroredStrategy()

spec = importlib.util.spec_from_file_location(LAYER_NAME + "_model", LAYER_NAME + "_model.py")
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


X_train = np.load(DATA_PATH + 'all_train_imgs.npy')
y_train = np.load(DATA_PATH + 'all_train_msks.npy')
X_val = np.load(DATA_PATH + 'all_val_imgs.npy')
y_val = np.load(DATA_PATH + 'all_val_msks.npy')


#-------------------------------------------------------------------------------------------------------------#

#Create validation set:
Xy_val = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(X_val),
                                tf.data.Dataset.from_tensor_slices(y_val))).cache().batch(BATCH_SIZE).prefetch(4)



# we create two instances with the same arguments
data_gen_args = dict(
    rotation_range=25.,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(X_train, seed=seed)
mask_datagen.fit(y_train, seed=seed)

image_generator = image_datagen.flow(X_train, batch_size=BATCH_SIZE, seed=seed)
mask_generator = mask_datagen.flow(y_train, batch_size=BATCH_SIZE, seed=seed)
train_generator = zip(image_generator, mask_generator)


#-------------------------------------------------------------------------------------------------------------#


# model = actual_unet(img_rows, img_cols)

# model.summary()
# #model_checkpoint = ModelCheckpoint(
# #    'drive/My Drive/weights_model1.h5', monitor='val_loss', save_best_only=True)

# model_checkpoint = ModelCheckpoint(
#    'weights_model1.h5', monitor='val_loss', save_best_only=True)

# lrate = LearningRateScheduler(step_decay)

# stopping = EarlyStopping(monitor='val_loss', patience=10)
# #model.load_weights('drive/My Drive/weights_model1.h5')
# model.compile(optimizer=Adam(), loss=weighted_binary_crossentropy,
#               metrics=[dice_coef, f1_m, precision_m, 'binary_accuracy', 'mae'])

# n_imgs = X_train.shape[0]

# model.fit_generator(
#     train_generator,
#     steps_per_epoch=sample_reps_per_epoch * n_imgs // batch_size,
#     epochs=5000,
#     verbose=1,
#     shuffle=True,
#     validation_data=(X_val, y_val),
#     callbacks=[model_checkpoint, lrate, stopping],
#     use_multiprocessing=True)



print("-" * 30)
print("Fitting model with training data ...")
print("-" * 30)

print("Training the model started at {}".format(datetime.datetime.now()))
start_time = time.time()

#Train the model
# model.load_weights(SAVE_PATH)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=SAVE_PATH,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

# history = model.fit(Xy_train,
#           validation_data=Xy_test,
#           epochs=EPOCHS, verbose=VERBOSE,
#           callbacks=[model_checkpoint_callback])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    shuffle=True,
    validation_data=Xy_val,
    callbacks=[model_checkpoint_callback],
    use_multiprocessing=True)


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