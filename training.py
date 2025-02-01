import os
import pathlib

import tensorflow as tf
import tensorflow_io as tfio 
import matplotlib.pyplot as plt
import numpy as np


DATASET_PATH = ""
data_dir = pathlib.Path(DATASET_PATH)

training_dataset = None
validation_dataset = None
validation_dataset = None
voice_recognition_Model = None
training_spectrogram_ds = None
validation_spectrogram_ds = None

def setDataSetPath(path):
    global DATASET_PATH, data_dir
    DATASET_PATH = path
    data_dir = pathlib.Path(DATASET_PATH)
    if not data_dir.exists():
        return False
    else:
        global training_dataset, validation_dataset
        training_dataset, validation_dataset = tf.keras.utils.audio_dataset_from_directory(
            directory=data_dir,
            batch_size=64, #specifies dataset should be loaded in batches of 64 audio samples
            validation_split=0.2, #20% of the data here will be used for self checking while the 80% will be used for data training
            seed=0,
            output_sequence_length=28000, #Hz that audio is being recorded in
            subset='both' #Defines whether the training subset is going to be used for training, validation or both. in this case both
        )
        #puts all the possible outputs (or "class" names into this one array)
        label_names = np.array(training_dataset.class_names)

        #shards = how to split it into. At index 0, this will be the first shard
        #What this does is it splits the data set in half and splits one half of the data set for testing
        #and the other for data validation for the neural network model
        #num shards = how to divide data set by. Index = the index
        test_dataset = validation_dataset.shard(num_shards = 2, index = 0)
        validation_dataset = validation_dataset.shard(num_shards = 2, index = 1)

        def get_spectrogram(waveform):
            #Converts waveform to a spectrogram via STFT
            spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
            #obtain magnitude of the STFT
            spectrogram = tf.abs(spectrogram)
            #Expects 4d input, so we create a new axis to combat this issue 
            spectrogram = spectrogram[..., tf.newaxis]
            return spectrogram


        def make_spectrogram_dataset(dataset): 
            return dataset.map(
                map_func=lambda audio, label: (get_spectrogram(audio), label), num_parallel_calls = tf.data.AUTOTUNE)
                #processes multiple elements of dataset at the same time, tunes automatically based on computer
        #Get the spectrograms of all the datasets
        global training_spectrogram_ds, validation_spectrogram_ds, test_spectrogram_ds
        training_spectrogram_ds = make_spectrogram_dataset(training_dataset)
        validation_spectrogram_ds = make_spectrogram_dataset(validation_dataset)
        test_spectrogram_ds = make_spectrogram_dataset(test_dataset)



        num_labels = len(label_names) #grabs the lenght of the label_names array
        #Instantiate normalization neural network layer
        norm_layer = tf.keras.Normalization()
        #Computes mean and variance of training data. Updates norm_layer to use this way
        norm_layer.adapt(data=training_spectrogram_ds.map(map_func=lambda spec, label: spec))

        for spectrogram, _ in training_spectrogram_ds.take(1):
            input_shape = spectrogram.shape

        #Define Sequential model
        global voice_recognition_Model 
        voice_recognition_Model =  tf.keras.Sequential([
            #Expected shape for input
            tf.keras.layers.Input(shape = input_shape),
            #downscales for easier processing
            tf.keras.layers.Resizing(32,32),
            #From Above
            norm_layer,
            #Uses 32 filters, each size 3x3 to extract features. 
            #ReLU is an activation function that outputs the input if it's positive. Otherwise zero
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            #Uses 64 filters, each size 3x3 to extract features. Even more detailed
            tf.keras.layers.Conv2D(64,3, activation='relu'),
            #Reduces size of spectrogram while keeping most important features
            tf.keras.layers.MaxPooling2D(),
            #This dropout prevents overlifting and becoming dependent on specific neurons. becomes more generalized
            tf.keras.layers.Dropout(0.25),
            #converts 2d feature maps into 1d so dense layer can read
            tf.keras.layers.Flatten(),
            #Very very very very detailed 
            tf.keras.layers.Dense(128, activation='relu'),
            #Drop 50% of neutrons during training
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_labels), #Number of outputs possible
        ])
        
        


#Number of runthroughs


def train(EPOCHS = 15):
    voice_recognition_Model.compile(
        #uses adaptive moment estimation optimizer, advanced version of stochastic gradient descent
        #This descent is an iterative method for optimizing an objective function. This is used in machine
        #learning and artificial intelligence to train models efficiently
        optimizer=tf.keras.optimizers.Adam(),
        #Loss function. Used for multiclass classification when labels are integers
        #Helps model adjust weights by calculating how wrong predictions are 
        #model outputs raw scores
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #Compares predicted class with correct label
        metrics=['accuracy'],
    )
    history = voice_recognition_Model.fit(
        #Set datasets
        training_spectrogram_ds,
        validation_data=validation_spectrogram_ds,
        epochs=EPOCHS,
        #patience = training will stop after x epochs
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )
    voice_recognition_Model.save('saved_model/my_model')

#voice_recognition_Model.evaluate()

#input layer
#voice_recognition_Model.add(tf.keras.layers.Input(shape = (128, ,10))
