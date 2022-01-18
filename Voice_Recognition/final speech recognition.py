from tensorflow import keras
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]= "3"
import pathlib
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import tensorflow as tf
import speech_recognition as sr
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
import pyaudio
import wave
import simpleaudio as sa
import time
import glob
import cv2
import speech_recognition as sr
import threading

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

data_dir = pathlib.Path('data/mini_speech_commands')
data_dir1 = pathlib.Path('data')


if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('Commands:', commands)
time.sleep(4)
"""
img=cv2.imread('white2.jpg')
cv2.imshow("Supported Commands", img)
cv2.waitKey(5000)
cv2.destroyAllWindows()
"""
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)

train_files = filenames[:7300]
val_files = filenames[7301:]
test_files = filenames[7301:]

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)

  # Note: You'll use indexing here instead of tuple unpacking to enable this 
  # to work in a TensorFlow graph.
  return parts[-2]

def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

def get_spectrogram(waveform):
  # Padding for files with less than 16000 samples
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)

  spectrogram = tf.abs(spectrogram)

  return spectrogram

for waveform, label in waveform_ds.take(1):
  label = label.numpy().decode('utf-8')
  spectrogram = get_spectrogram(waveform)

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(label == commands)
  return spectrogram, label_id

spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds

train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape
#print('Input shape:', input_shape)
num_labels = len(commands)

norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

#model = keras.models.load_model('save_model_speech_recog7.h5') #7 khoobe
model = keras.models.load_model('save_model_speech_recog_new_dataset3.h5') #7 khoobe

r = sr.Recognizer()
sr.Microphone()
##########################################

def wave_maker1():
    while True:
        start=time.perf_counter()
        WAVE_OUTPUT_FILENAME1 = "data/rec/1.wav"
        CHUNK1 = 1024
        FORMAT1 = pyaudio.paInt16
        CHANNELS1 = 1
        RATE1 = 16000
        RECORD_SECONDS1 =1
        p1 = pyaudio.PyAudio()
        stream1 = p1.open(format=FORMAT1,
                    channels=CHANNELS1,
                    rate=RATE1,
                    input=True,
                    frames_per_buffer=CHUNK1)
        #finish=time.perf_counter()
        #pt1=finish-start
        #print('pt1=',pt1)
        print("* recording1")
        frames1 = []
        for i in range(0, int(RATE1 / CHUNK1 * RECORD_SECONDS1)):
            data1 = stream1.read(CHUNK1)
            frames1.append(data1)
        stream1.stop_stream()
        stream1.close()
        p1.terminate()
        wf1 = wave.open(WAVE_OUTPUT_FILENAME1, 'wb')
        wf1.setnchannels(CHANNELS1)
        wf1.setsampwidth(p1.get_sample_size(FORMAT1))
        wf1.setframerate(RATE1)
        wf1.writeframes(b''.join(frames1))
        wf1.close()
        finish=time.perf_counter()
        #duration=finish-start
        #print('duration=',duration)
        duration=finish-start
        time.sleep(1.5-duration)
        finish2=time.perf_counter()
        duration=finish2-start
        #print('duration1=',duration)

def wave_maker2():
    while True:
        start=time.perf_counter()
        WAVE_OUTPUT_FILENAME1 = "data/rec/2.wav"
        CHUNK1 = 1024
        FORMAT1 = pyaudio.paInt16
        CHANNELS1 = 1
        RATE1 = 16000
        RECORD_SECONDS1 =1
        p1 = pyaudio.PyAudio()
        stream1 = p1.open(format=FORMAT1,
                    channels=CHANNELS1,
                    rate=RATE1,
                    input=True,
                    frames_per_buffer=CHUNK1)

        print("* recording2")
        frames1 = []
        for i in range(0, int(RATE1 / CHUNK1 * RECORD_SECONDS1)):
            data1 = stream1.read(CHUNK1)
            frames1.append(data1)
        stream1.stop_stream()
        stream1.close()
        p1.terminate()
        wf1 = wave.open(WAVE_OUTPUT_FILENAME1, 'wb')
        wf1.setnchannels(CHANNELS1)
        wf1.setsampwidth(p1.get_sample_size(FORMAT1))
        wf1.setframerate(RATE1)
        wf1.writeframes(b''.join(frames1))
        wf1.close()
        finish=time.perf_counter()
        #duration=finish-start
        #print('duration=',duration)
        duration=finish-start
        time.sleep(1.5-duration)
        finish2=time.perf_counter()
        duration=finish2-start
        #print('duration2=',duration)

def wave_maker3():
    while True:
        start=time.perf_counter()
        WAVE_OUTPUT_FILENAME1 = "data/rec/3.wav"
        CHUNK1 = 1024
        FORMAT1 = pyaudio.paInt16
        CHANNELS1 = 1
        RATE1 = 16000
        RECORD_SECONDS1 =1
        p1 = pyaudio.PyAudio()
        stream1 = p1.open(format=FORMAT1,
                    channels=CHANNELS1,
                    rate=RATE1,
                    input=True,
                    frames_per_buffer=CHUNK1)

        print("* recording3")
        frames1 = []
        for i in range(0, int(RATE1 / CHUNK1 * RECORD_SECONDS1)):
            data1 = stream1.read(CHUNK1)
            frames1.append(data1)
        stream1.stop_stream()
        stream1.close()
        p1.terminate()
        wf1 = wave.open(WAVE_OUTPUT_FILENAME1, 'wb')
        wf1.setnchannels(CHANNELS1)
        wf1.setsampwidth(p1.get_sample_size(FORMAT1))
        wf1.setframerate(RATE1)
        wf1.writeframes(b''.join(frames1))
        wf1.close()
        finish=time.perf_counter()
        #duration=finish-start
        #print('duration=',duration)
        duration=finish-start
        time.sleep(1.5-duration)
        finish2=time.perf_counter()
        duration=finish2-start
        #print('duration3=',duration)

def wave_maker4():
    while True:
        start=time.perf_counter()
        WAVE_OUTPUT_FILENAME1 = "data/rec/4.wav"
        CHUNK1 = 1024
        FORMAT1 = pyaudio.paInt16
        CHANNELS1 = 1
        RATE1 = 16000
        RECORD_SECONDS1 =1
        p1 = pyaudio.PyAudio()
        stream1 = p1.open(format=FORMAT1,
                    channels=CHANNELS1,
                    rate=RATE1,
                    input=True,
                    frames_per_buffer=CHUNK1)

        print("* recording4")
        frames1 = []
        for i in range(0, int(RATE1 / CHUNK1 * RECORD_SECONDS1)):
            data1 = stream1.read(CHUNK1)
            frames1.append(data1)
        stream1.stop_stream()
        stream1.close()
        p1.terminate()
        wf1 = wave.open(WAVE_OUTPUT_FILENAME1, 'wb')
        wf1.setnchannels(CHANNELS1)
        wf1.setsampwidth(p1.get_sample_size(FORMAT1))
        wf1.setframerate(RATE1)
        wf1.writeframes(b''.join(frames1))
        wf1.close()
        finish=time.perf_counter()
        #duration=finish-start
        #print('duration=',duration)
        duration=finish-start
        time.sleep(1.5-duration)
        finish2=time.perf_counter()
        duration=finish2-start
        #print('duration4=',duration)

def wave_maker5():
    while True:
        start=time.perf_counter()
        WAVE_OUTPUT_FILENAME1 = "data/rec/5.wav"
        CHUNK1 = 1024
        FORMAT1 = pyaudio.paInt16
        CHANNELS1 = 1
        RATE1 = 16000
        RECORD_SECONDS1 =1
        p1 = pyaudio.PyAudio()
        stream1 = p1.open(format=FORMAT1,
                    channels=CHANNELS1,
                    rate=RATE1,
                    input=True,
                    frames_per_buffer=CHUNK1)

        print("* recording5")
        frames1 = []
        for i in range(0, int(RATE1 / CHUNK1 * RECORD_SECONDS1)):
            data1 = stream1.read(CHUNK1)
            frames1.append(data1)
        stream1.stop_stream()
        stream1.close()
        p1.terminate()
        wf1 = wave.open(WAVE_OUTPUT_FILENAME1, 'wb')
        wf1.setnchannels(CHANNELS1)
        wf1.setsampwidth(p1.get_sample_size(FORMAT1))
        wf1.setframerate(RATE1)
        wf1.writeframes(b''.join(frames1))
        wf1.close()
        finish=time.perf_counter()
        #duration=finish-start
        #print('duration=',duration)
        duration=finish-start
        time.sleep(1.5-duration)
        finish2=time.perf_counter()
        duration=finish2-start
        #print('duration5=',duration)

def wave_maker6():
    while True:
        start=time.perf_counter()
        WAVE_OUTPUT_FILENAME1 = "data/rec/6.wav"
        CHUNK1 = 1024
        FORMAT1 = pyaudio.paInt16
        CHANNELS1 = 1
        RATE1 = 16000
        RECORD_SECONDS1 =1
        p1 = pyaudio.PyAudio()
        stream1 = p1.open(format=FORMAT1,
                    channels=CHANNELS1,
                    rate=RATE1,
                    input=True,
                    frames_per_buffer=CHUNK1)

        print("* recording6")
        frames1 = []
        for i in range(0, int(RATE1 / CHUNK1 * RECORD_SECONDS1)):
            data1 = stream1.read(CHUNK1)
            frames1.append(data1)
        stream1.stop_stream()
        stream1.close()
        p1.terminate()
        wf1 = wave.open(WAVE_OUTPUT_FILENAME1, 'wb')
        wf1.setnchannels(CHANNELS1)
        wf1.setsampwidth(p1.get_sample_size(FORMAT1))
        wf1.setframerate(RATE1)
        wf1.writeframes(b''.join(frames1))
        wf1.close()
        finish=time.perf_counter()
        #duration=finish-start
        #print('duration=',duration)
        duration=finish-start
        time.sleep(1.5-duration)
        finish2=time.perf_counter()
        duration=finish2-start
        #print('duration6=',duration)

def wave_maker7():
    while True:
        start=time.perf_counter()
        WAVE_OUTPUT_FILENAME1 = "data/rec/7.wav"
        CHUNK1 = 1024
        FORMAT1 = pyaudio.paInt16
        CHANNELS1 = 1
        RATE1 = 16000
        RECORD_SECONDS1 =1
        p1 = pyaudio.PyAudio()
        stream1 = p1.open(format=FORMAT1,
                    channels=CHANNELS1,
                    rate=RATE1,
                    input=True,
                    frames_per_buffer=CHUNK1)

        print("* recording7")
        frames1 = []
        for i in range(0, int(RATE1 / CHUNK1 * RECORD_SECONDS1)):
            data1 = stream1.read(CHUNK1)
            frames1.append(data1)
        stream1.stop_stream()
        stream1.close()
        p1.terminate()
        wf1 = wave.open(WAVE_OUTPUT_FILENAME1, 'wb')
        wf1.setnchannels(CHANNELS1)
        wf1.setsampwidth(p1.get_sample_size(FORMAT1))
        wf1.setframerate(RATE1)
        wf1.writeframes(b''.join(frames1))
        wf1.close()
        finish=time.perf_counter()
        #duration=finish-start
        #print('duration=',duration)
        duration=finish-start
        time.sleep(1.5-duration)
        finish2=time.perf_counter()
        duration=finish2-start
        #print('duration7=',duration)

def wave_maker8():
    while True:
        start=time.perf_counter()
        WAVE_OUTPUT_FILENAME1 = "data/rec/8.wav"
        CHUNK1 = 1024
        FORMAT1 = pyaudio.paInt16
        CHANNELS1 = 1
        RATE1 = 16000
        RECORD_SECONDS1 =1
        p1 = pyaudio.PyAudio()
        stream1 = p1.open(format=FORMAT1,
                    channels=CHANNELS1,
                    rate=RATE1,
                    input=True,
                    frames_per_buffer=CHUNK1)

        print("* recording8")
        frames1 = []
        for i in range(0, int(RATE1 / CHUNK1 * RECORD_SECONDS1)):
            data1 = stream1.read(CHUNK1)
            frames1.append(data1)
        stream1.stop_stream()
        stream1.close()
        p1.terminate()
        wf1 = wave.open(WAVE_OUTPUT_FILENAME1, 'wb')
        wf1.setnchannels(CHANNELS1)
        wf1.setsampwidth(p1.get_sample_size(FORMAT1))
        wf1.setframerate(RATE1)
        wf1.writeframes(b''.join(frames1))
        wf1.close()
        finish=time.perf_counter()
        #duration=finish-start
        #print('duration=',duration)
        duration=finish-start
        time.sleep(1.5-duration)
        finish2=time.perf_counter()
        duration=finish2-start
        #print('duration8=',duration)

def wave_maker9():
    while True:
        start=time.perf_counter()
        WAVE_OUTPUT_FILENAME1 = "data/rec/9.wav"
        CHUNK1 = 1024
        FORMAT1 = pyaudio.paInt16
        CHANNELS1 = 1
        RATE1 = 16000
        RECORD_SECONDS1 =1
        p1 = pyaudio.PyAudio()
        stream1 = p1.open(format=FORMAT1,
                    channels=CHANNELS1,
                    rate=RATE1,
                    input=True,
                    frames_per_buffer=CHUNK1)

        print("* recording3")
        frames1 = []
        for i in range(0, int(RATE1 / CHUNK1 * RECORD_SECONDS1)):
            data1 = stream1.read(CHUNK1)
            frames1.append(data1)
        stream1.stop_stream()
        stream1.close()
        p1.terminate()
        wf1 = wave.open(WAVE_OUTPUT_FILENAME1, 'wb')
        wf1.setnchannels(CHANNELS1)
        wf1.setsampwidth(p1.get_sample_size(FORMAT1))
        wf1.setframerate(RATE1)
        wf1.writeframes(b''.join(frames1))
        wf1.close()
        finish=time.perf_counter()
        #duration=finish-start
        #print('duration=',duration)
        duration=finish-start
        time.sleep(1.5-duration)
        finish2=time.perf_counter()
        duration=finish2-start
        #print('duration9=',duration)

def wave_maker10():
    while True:
        start=time.perf_counter()
        WAVE_OUTPUT_FILENAME1 = "data/rec/10.wav"
        CHUNK1 = 1024
        FORMAT1 = pyaudio.paInt16
        CHANNELS1 = 1
        RATE1 = 16000
        RECORD_SECONDS1 =1
        p1 = pyaudio.PyAudio()
        stream1 = p1.open(format=FORMAT1,
                    channels=CHANNELS1,
                    rate=RATE1,
                    input=True,
                    frames_per_buffer=CHUNK1)

        print("* recording10")
        frames1 = []
        for i in range(0, int(RATE1 / CHUNK1 * RECORD_SECONDS1)):
            data1 = stream1.read(CHUNK1)
            frames1.append(data1)
        stream1.stop_stream()
        stream1.close()
        p1.terminate()
        wf1 = wave.open(WAVE_OUTPUT_FILENAME1, 'wb')
        wf1.setnchannels(CHANNELS1)
        wf1.setsampwidth(p1.get_sample_size(FORMAT1))
        wf1.setframerate(RATE1)
        wf1.writeframes(b''.join(frames1))
        wf1.close()
        finish=time.perf_counter()
        #duration=finish-start
        #print('duration=',duration)
        duration=finish-start
        time.sleep(1.5-duration)
        finish2=time.perf_counter()
        duration=finish2-start
        #print('duration10=',duration)

def wave_maker11():
    while True:
        start=time.perf_counter()
        WAVE_OUTPUT_FILENAME1 = "data/rec/11.wav"
        CHUNK1 = 1024
        FORMAT1 = pyaudio.paInt16
        CHANNELS1 = 1
        RATE1 = 16000
        RECORD_SECONDS1 =1
        p1 = pyaudio.PyAudio()
        stream1 = p1.open(format=FORMAT1,
                    channels=CHANNELS1,
                    rate=RATE1,
                    input=True,
                    frames_per_buffer=CHUNK1)

        print("* recording11")
        frames1 = []
        for i in range(0, int(RATE1 / CHUNK1 * RECORD_SECONDS1)):
            data1 = stream1.read(CHUNK1)
            frames1.append(data1)
        stream1.stop_stream()
        stream1.close()
        p1.terminate()
        wf1 = wave.open(WAVE_OUTPUT_FILENAME1, 'wb')
        wf1.setnchannels(CHANNELS1)
        wf1.setsampwidth(p1.get_sample_size(FORMAT1))
        wf1.setframerate(RATE1)
        wf1.writeframes(b''.join(frames1))
        wf1.close()
        finish=time.perf_counter()
        #duration=finish-start
        #print('duration=',duration)
        duration=finish-start
        time.sleep(1.5-duration)
        finish2=time.perf_counter()
        duration=finish2-start
        #print('duration11=',duration)

def wave_maker12():
    while True:
        start=time.perf_counter()
        WAVE_OUTPUT_FILENAME1 = "data/rec/12.wav"
        CHUNK1 = 1024
        FORMAT1 = pyaudio.paInt16
        CHANNELS1 = 1
        RATE1 = 16000
        RECORD_SECONDS1 =1
        p1 = pyaudio.PyAudio()
        stream1 = p1.open(format=FORMAT1,
                    channels=CHANNELS1,
                    rate=RATE1,
                    input=True,
                    frames_per_buffer=CHUNK1)

        print("* recording12")
        frames1 = []
        for i in range(0, int(RATE1 / CHUNK1 * RECORD_SECONDS1)):
            data1 = stream1.read(CHUNK1)
            frames1.append(data1)
        stream1.stop_stream()
        stream1.close()
        p1.terminate()
        wf1 = wave.open(WAVE_OUTPUT_FILENAME1, 'wb')
        wf1.setnchannels(CHANNELS1)
        wf1.setsampwidth(p1.get_sample_size(FORMAT1))
        wf1.setframerate(RATE1)
        wf1.writeframes(b''.join(frames1))
        wf1.close()
        finish=time.perf_counter()
        #duration=finish-start
        #print('duration=',duration)
        duration=finish-start
        time.sleep(1.5-duration)
        finish2=time.perf_counter()
        duration=finish2-start
        #print('duration12=',duration)

def wave_maker13():
    while True:
        start=time.perf_counter()
        WAVE_OUTPUT_FILENAME1 = "data/rec/13.wav"
        CHUNK1 = 1024
        FORMAT1 = pyaudio.paInt16
        CHANNELS1 = 1
        RATE1 = 16000
        RECORD_SECONDS1 =1
        p1 = pyaudio.PyAudio()
        stream1 = p1.open(format=FORMAT1,
                    channels=CHANNELS1,
                    rate=RATE1,
                    input=True,
                    frames_per_buffer=CHUNK1)

        print("* recording13")
        frames1 = []
        for i in range(0, int(RATE1 / CHUNK1 * RECORD_SECONDS1)):
            data1 = stream1.read(CHUNK1)
            frames1.append(data1)
        stream1.stop_stream()
        stream1.close()
        p1.terminate()
        wf1 = wave.open(WAVE_OUTPUT_FILENAME1, 'wb')
        wf1.setnchannels(CHANNELS1)
        wf1.setsampwidth(p1.get_sample_size(FORMAT1))
        wf1.setframerate(RATE1)
        wf1.writeframes(b''.join(frames1))
        wf1.close()
        finish=time.perf_counter()
        #duration=finish-start
        #print('duration=',duration)
        duration=finish-start
        time.sleep(1.5-duration)
        finish2=time.perf_counter()
        duration=finish2-start
        #print('duration13=',duration)

def wave_maker14():
    while True:
        start=time.perf_counter()
        WAVE_OUTPUT_FILENAME1 = "data/rec/14.wav"
        CHUNK1 = 1024
        FORMAT1 = pyaudio.paInt16
        CHANNELS1 = 1
        RATE1 = 16000
        RECORD_SECONDS1 =1
        p1 = pyaudio.PyAudio()
        stream1 = p1.open(format=FORMAT1,
                    channels=CHANNELS1,
                    rate=RATE1,
                    input=True,
                    frames_per_buffer=CHUNK1)

        print("* recording14")
        frames1 = []
        for i in range(0, int(RATE1 / CHUNK1 * RECORD_SECONDS1)):
            data1 = stream1.read(CHUNK1)
            frames1.append(data1)
        stream1.stop_stream()
        stream1.close()
        p1.terminate()
        wf1 = wave.open(WAVE_OUTPUT_FILENAME1, 'wb')
        wf1.setnchannels(CHANNELS1)
        wf1.setsampwidth(p1.get_sample_size(FORMAT1))
        wf1.setframerate(RATE1)
        wf1.writeframes(b''.join(frames1))
        wf1.close()
        finish=time.perf_counter()
        #duration=finish-start
        #print('duration=',duration)
        duration=finish-start
        time.sleep(1.5-duration)
        finish2=time.perf_counter()
        duration=finish2-start
        #print('duration14=',duration)

def talk():
    t1=threading.Thread(target=wave_maker1)
    t2=threading.Thread(target=wave_maker2)
    t3=threading.Thread(target=wave_maker3)
    t4=threading.Thread(target=wave_maker4)
    t5=threading.Thread(target=wave_maker5)
    t6=threading.Thread(target=wave_maker6)
    t7=threading.Thread(target=wave_maker7)
    t8=threading.Thread(target=wave_maker8)
    t9=threading.Thread(target=wave_maker9)
    t10=threading.Thread(target=wave_maker10)
    t11=threading.Thread(target=wave_maker11)
    t12=threading.Thread(target=wave_maker12)
    t13=threading.Thread(target=wave_maker13)
    t14=threading.Thread(target=wave_maker14)


    t1.start()
    time.sleep(0.1)
    t2.start()
    time.sleep(0.1)
    t3.start()
    time.sleep(0.1)
    t4.start()
    time.sleep(0.1)
    t5.start()
    time.sleep(0.1)
    t6.start()
    time.sleep(0.1)
    t7.start()
    time.sleep(0.1)
    t8.start()
    time.sleep(0.1)
    t9.start()
    time.sleep(0.1)
    t10.start()
    time.sleep(0.1)
    t11.start()
    time.sleep(0.1)
    t12.start()
    time.sleep(0.1)
    t13.start()
    time.sleep(0.1)
    t14.start()
    time.sleep(0.1)
    recorded=['rec/1.wav','rec/2.wav','rec/3.wav','rec/4.wav','rec/5.wav',
    'rec/6.wav','rec/7.wav','rec/8.wav','rec/9.wav','rec/10.wav',
    'rec/11.wav','rec/12.wav','rec/13.wav','rec/14.wav']
    #########################################
    i=-1
    while True:
      i=i+1
      if i==13:
        i=-1
      #with plt.ion():
      sample_file =data_dir1/recorded[0]
      sample_ds = preprocess_dataset([str(sample_file)])
      for spectrogram, label in sample_ds.batch(1):
          prediction = model(spectrogram)

          codes= tf.nn.softmax(prediction[0])
          codes=np.array(codes)
          max_value=np.max(codes)
          #print(codes)
          max_value_arg = np.argmax(codes)
          #print(max_value)
          if max_value > 0.9:
              predicted_command=commands[max_value_arg]

              print(predicted_command)
              print(type(predicted_command))
              return predicted_command





while 1:
    print(talk())