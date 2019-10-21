import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
from nltk.tokenize import RegexpTokenizer
from keras.models import load_model
import numpy as np
import json
from textwrap import wrap
import re

def keras_analyze(text_columns):

    all_sent_values = []
    # paths to deep learning model and word index
    path = '/Users/chandan/MachineLearningProject/sentimentalAnalysis/elmoTest'
    weight_path = path + '/model/best_model.hdf5'

    word_idx = word_idx = json.load(open(path + '/Data/word_idx.txt'))
    from keras import backend as k
    with tf.Session() as session:
        k.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        loaded_model = load_model(weight_path)

        for i in range(0, text_columns.size):
            count = 0
            count = len(re.findall(r'\w+', str(text_columns[i])))
            if count > 56:
                string = text_columns[i]
                formatted = wrap(string, 180)
                tot_score = 0.00
                for k in range(0, len(formatted)):
                    tot_score = tot_score + \
                        (live_test(loaded_model, str(formatted[k]), word_idx))
                    chunks = len(formatted)
                avg_score = (tot_score/chunks)
                avg_score = round(avg_score, 2)
                all_sent_values.append(avg_score)
            else:
                all_sent_values.append(
                    live_test(loaded_model, str(text_columns[i]), word_idx))
        session.close()
    return all_sent_values

def live_test(trained_model, data, word_idx):
    live_list = []
    live_list_np = np.zeros((56, 1))

    # split the sentence into its words and remove any punctuations.

    tokenizer = RegexpTokenizer(r'\w+')

    data_sample_list = tokenizer.tokenize(data)

    # get word index and the respective embeddings

    data_index = np.array([word_idx[word.lower()] if word.lower(
    ) in word_idx else 0 for word in data_sample_list])

    data_index_np = np.array(data_index)

    # padded with zeros of length 56 maximum word count

    padded_array = np.zeros(56)

    padded_array[:data_index_np.shape[0]] = data_index_np

    data_index_np_pad = padded_array.astype(int)

    live_list.append(data_index_np_pad)

    live_list_np = np.asarray(live_list)

    # get score from the model

    score = trained_model.predict(live_list_np, batch_size=1, verbose=0)

    # maximum of the array, unused
    single_score = np.round(np.argmax(score)/10, decimals=2)

    # weighted score of top 3 bands

    top_3_index = np.argsort(score)[0][-3:]

    top_3_scores = score[0][top_3_index]

    top_3_weights = top_3_scores/np.sum(top_3_scores)

    single_score_dot = np.round(
        np.dot(top_3_index, top_3_weights)/10, decimals=2)

    return single_score_dot
