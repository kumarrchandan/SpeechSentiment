# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the streaming API.
NOTE: This module requires the additional dependency `pyaudio`. To install
using pip:
    pip install pyaudio
Example usage:
    python transcribe_streaming_indefinite.py
"""

# [START speech_transcribe_infinite_streaming]
from __future__ import division

import time
import re
import sys
import os
import spacy
from google.cloud import speech

#from sampleElmoImpl import clean_data
import pyaudio
from six.moves import queue

import keras
from keras import backend as k
from keras.models import Model

from keras.layers import Dense,Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import History
from keras.layers import Input, Dense
import tensorflow_hub as hub

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

# Audio recording parameters
STREAMING_LIMIT = 290000
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

os.environ['GOOGLE_APPLICATION_CREDENTIALS']="/Users/chandan/GoogleServiceAccountKey/serviceACKey.json"

def get_current_time():
    return int(round(time.time() * 1000))


def duration_to_secs(duration):
    return duration.seconds + (duration.nanos / float(1e9))


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk_size):
        self._rate = rate
        self._chunk_size = chunk_size
        self._num_channels = 1
        self._max_replay_secs = 5

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()

        # 2 bytes in 16 bit samples
        self._bytes_per_sample = 2 * self._num_channels
        self._bytes_per_second = self._rate * self._bytes_per_sample

        self._bytes_per_chunk = (self._chunk_size * self._bytes_per_sample)
        self._chunks_per_second = (
                self._bytes_per_second // self._bytes_per_chunk)

    def __enter__(self):
        self.closed = False

        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            if get_current_time() - self.start_time > STREAMING_LIMIT:
                self.start_time = get_current_time()
                break
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)


def listen_print_loop(responses, stream):
    """Iterates through server responses and prints them.
    The responses passed is a generator that will block until a response
    is provided by the server.
    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.
    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    responses = (r for r in responses if (
            r.results and r.results[0].alternatives))

    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        top_alternative = result.alternatives[0]
        transcript = top_alternative.transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

#        if not result.is_final:
#            sys.stdout.write(transcript + overwrite_chars + '\r')
#            sys.stdout.flush()
#
#            num_chars_printed = len(transcript)
#        else:
        if (result.is_final):
            print(transcript + overwrite_chars)
#            reRunSavedModel('Pass Your String Here')
            # Exit recognition if any of the transcribed phrases could be one of our keywords.
            if re.search(r'\b(stop|exit|quit)\b', transcript, re.I):
                print('Exiting..')
                stream.closed = True
                print(transcript + overwrite_chars)
                break

            num_chars_printed = 0


def main():
    client = speech.SpeechClient()
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code='en-US',
        max_alternatives=1,
        enable_word_time_offsets=True)
    streaming_config = speech.types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)

    print('Say "Stop","Quit" or "Exit" to terminate the program.')

    with mic_manager as stream:
        while not stream.closed:
            audio_generator = stream.generator()
            requests = (speech.types.StreamingRecognizeRequest(
                audio_content=content)
                for content in audio_generator)

            responses = client.streaming_recognize(streaming_config,
                                                   requests)
            # Now, put the transcription responses to use.
            listen_print_loop(responses, stream)

def build_model():
    input_text = Input(shape=(1024,), dtype=tf.float32)
    dense = Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(input_text)
    pred = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=[input_text], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def clean_data(xyzOrig):

    punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
    xyz = ''.join (ch for ch in xyzOrig if ch not in set(punctuation))
    xyz=xyz.lower()
    xyz = xyz.strip('0123456789')
    xyz = ' '.join(xyz.split())
    xyz = lemmatization([xyz])

    emb = elmo(xyz, signature="default", as_dict=True)["elmo"]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        elmo_my = sess.run(tf.reduce_mean(emb,1))
    return elmo_my

def reRunSavedModel(strText):
    with tf.Session() as session:
        k.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        model_elmoChan = build_model()
        model_elmoChan.load_weights('model_elmo_chandan_weights.h5')
        import time
        t = time.time()
        cleanText = clean_data(strText)
        predicts = model_elmoChan.predict(cleanText)
        print("time: ", time.time() - t)
        print(predicts)

def lemmatization(texts):
    nlp = spacy.load('en', disable=['parser', 'ner'])
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output

#reRunSavedModel('Fruit just tastes better when you pick it yourself.')
if __name__ == '__main__':
    main()
# [END speech_transcribe_infinite_streaming]

