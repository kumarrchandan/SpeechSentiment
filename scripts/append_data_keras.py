def keras_append(all_sent_values, data_df):

    SENTIMENT_VALUE = []

    SENTIMENT = []

    for i in range(0, len(all_sent_values)):

        sent = all_sent_values[i]

        if (sent <= 1 and sent >= 0.75):

            SENTIMENT.append('Very Positive')

            SENTIMENT_VALUE.append(sent)

        elif (sent < 0.75 and sent >= 0.55):

            SENTIMENT.append('Positive')

            SENTIMENT_VALUE.append(sent)

        elif (sent < 0.55 and sent >= 0.45):

            SENTIMENT.append('Neutral')

            SENTIMENT_VALUE.append(sent)

        elif (sent < 0.45 and sent >= 0.25):

            SENTIMENT.append('Negative')

            SENTIMENT_VALUE.append(sent)

        else:

            SENTIMENT.append('Very Negative')

            SENTIMENT_VALUE.append(sent)

    temp_data = data_df

#    temp_data['Deep Model Score'] = SENTIMENT_VALUE

    temp_data['Deep Model Sentiment|'] = SENTIMENT

    return temp_data
