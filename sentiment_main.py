from sentimentalAnalysis.elmoTest.scripts.analyze_sentiment_keras import keras_analyze
from sentimentalAnalysis.elmoTest.scripts.analyze_sentiment_vader import vader_analyze
from sentimentalAnalysis.elmoTest.scripts.get_data import fetch, throw
from sentimentalAnalysis.elmoTest.scripts.append_data_keras import keras_append
from sentimentalAnalysis.elmoTest.scripts.append_data_vader import vader_append
from keras.models import load_model
import pandas as pd
#from sentimentalAnalysis.elmoTest.scripts.analyze_sentiment_keras import text_kerasAnalyze
#from sentimentalAnalysis.elmoTest.scripts.append_data_keras import kerasText_append

def sentiment(transcript):
    dataset_df = pd.DataFrame()
    transcript_df = []
    transcript_df.append(transcript)
    text_column_df = pd.DataFrame(transcript_df)
#    print(kerasText_append(text_kerasAnalyze(text_column_df)))
    with_sentiment_df = keras_append(keras_analyze(text_column_df), dataset_df)
    with_all_sentiment_df = vader_append(vader_analyze(text_column_df), with_sentiment_df)
    print(transcript,'\n',with_all_sentiment_df)

#    print(with_all_sentiment_df)

#def sentiment(transcript):
#    dataset_df = pd.DataFrame()
#    transcript_df = []
#    transcript_df.append(transcript)
#    text_column_df = pd.DataFrame(transcript_df)
#    with_sentiment_df = keras_append(keras_analyze(text_column_df), dataset_df)
#    with_all_sentiment_df = vader_append(vader_analyze(text_column_df), with_sentiment_df)
#    print('\n''\n')
#    print(transcript)
#    print(with_all_sentiment_df)
#
#sentiment('Fruit just tastes better when you pick it yourself.')