import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


DATASET_PATH = ""
data_dir = pathlib.Path(DATASET_PATH)

training_dataset = None
validation_dataset = None
voice_recognition_Model = None
training_spectrogram_ds = None
validation_spectrogram_ds = None
label_names = None

import os


def setDataSetPath(path):
    global DATASET_PATH, data_dir
    DATASET_PATH = path
    data_dir = pathlib.Path(DATASET_PATH)
    if not data_dir.exists():
        return False
    else:
        global training_dataset, validation_dataset, voice_recognition_Model, training_spectrogram_ds, validation_spectrogram_ds
        training_dataset, validation_dataset = tf.keras.utils.audio_dataset_from_directory(
            directory=data_dir,
            batch_size=64,
            validation_split=0.2,
            seed=0,
            output_sequence_length=384000,
            subset='both')

        #puts all the possible outputs (or "class" names into this one array)
        global label_names
        label_names = np.array(training_dataset.class_names)

        #shards = how to split it into. At index 0, this will be the first shard
        #What this does is it splits the data set in half and splits one half of the data set for testing
        #and the other for data validation for the neural network model
        #num shards = how to divide data set by. Index = the index
        def squeeze(audio, labels):
            audio = tf.squeeze(audio, axis=-1)
            return audio, labels
        
        training_dataset = training_dataset.map(squeeze, tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.map(squeeze, tf.data.AUTOTUNE)

        test_dataset = validation_dataset.shard(num_shards = 2, index = 0)
        validation_dataset = validation_dataset.shard(num_shards = 2, index = 1)

        for example_audio, example_labels in training_dataset.take(1):  
            print(example_audio.shape)
            print(example_labels.shape)


        def get_spectrogram(waveform):
            spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
            spectrogram = tf.abs(spectrogram)
            spectrogram = spectrogram[..., tf.newaxis]

            return spectrogram


        for i in range(3):
            label = label_names[example_labels[i]]
            waveform = example_audio[i]
            spectrogram = get_spectrogram(waveform)

            print('Label:', label)
            print('Waveform shape:', waveform.shape)
            print('Spectrogram shape:', spectrogram.shape)

        def make_spectrogram_dataset(dataset): 
            return dataset.map(
                map_func=lambda audio, label: (get_spectrogram(audio), label), num_parallel_calls = tf.data.AUTOTUNE)
                #processes multiple elements of dataset at the same time, tunes automatically based on computer
        #Get the spectrograms of all the datasets

        training_spectrogram_ds = make_spectrogram_dataset(training_dataset)
        validation_spectrogram_ds = make_spectrogram_dataset(validation_dataset)
        test_spectrogram_ds = make_spectrogram_dataset(test_dataset)


        for example_spectrograms in training_spectrogram_ds.take(1):
            break

        input_shape = example_spectrograms[0].shape[1:]
        print('Input shape:', input_shape)
        num_labels = len(label_names)

        # Instantiate the `tf.keras.layers.Normalization` layer.
        norm_layer = tf.keras.layers.Normalization()
        # Fit the state of the layer to the spectrograms
        # with `Normalization.adapt`.
        norm_layer.adapt(data=training_spectrogram_ds.map(map_func=lambda spec, label: spec))

        voice_recognition_Model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            # Downsample the input.
            tf.keras.layers.Resizing(32, 32),
            # Normalize.
            norm_layer,
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_labels),
        ])

        voice_recognition_Model.summary()


        return True
        
        


#Number of runthroughs


def train(EPOCHS = 15):
    global voice_recognition_Model
    global training_spectrogram_ds
    global validation_spectrogram_ds
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
    voice_recognition_Model.save('/Users/fatblabs/GitHub Repos/audio-recognition.keras')

def evaluate(sampleEvaluatePath):
    def get_spectrogram(waveform):
                spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
                spectrogram = tf.abs(spectrogram)
                spectrogram = spectrogram[..., tf.newaxis]

                return spectrogram
    voice_recognition_Model = tf.keras.models.load_model('/Users/fatblabs/GitHub Repos/audio-recognition.keras')
    x = tf.io.read_file(sampleEvaluatePath)
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
    x = tf.squeeze(x, axis=-1)
    waveform = x
    x = get_spectrogram(x)
    x = x[tf.newaxis,...]



    prediction = voice_recognition_Model(x)
    predicted_class = np.argmax(prediction, axis=-1)

    print(f"Prediction: {predicted_class}")
    return label_names[predicted_class]

#setDataSetPath("/Users/fatblabs/GitHub Repos/audio-recognition/DataSet")
#train()
#print(evaluate("/Users/fatblabs/Downloads/testingAudio.wav"))
#voice_recognition_Model.evaluate()
#/Users/fatblabs/Downloads/testingfinal.wav"
#/Users/fatblabs/GitHub Repos/audio-recognition/DataSet
#input layer
#voice_recognition_Model.add(tf.keras.layers.Input(shape = (128, ,10))
