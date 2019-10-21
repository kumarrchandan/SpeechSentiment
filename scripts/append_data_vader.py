def vader_append(all_sent_values, data_df):

    SENTIMENT_VALUE = []

    SENTIMENT = []

    #sorts the raw sentiment values into 5 categories

    for i in range(0, len(all_sent_values)):

        sent = all_sent_values[i]

        if (sent <= 1 and sent >= 0.5):

            SENTIMENT.append('Very Positive')

            SENTIMENT_VALUE.append(sent)

        elif (sent < 0.5 and sent >= 0.1):

            SENTIMENT.append('Positive')

            SENTIMENT_VALUE.append(sent)

        elif (sent < 0.1 and sent >= -0.1):

            SENTIMENT.append('Neutral')

            SENTIMENT_VALUE.append(sent)

        elif (sent < -0.1 and sent >= -0.5):

            SENTIMENT.append('Negative')

            SENTIMENT_VALUE.append(sent)

        else:

            SENTIMENT.append('Very Negative')

            SENTIMENT_VALUE.append(sent)

    #two columns with raw score and the sentiment category

    temp_data = data_df

#    temp_data['Vader Score'] = SENTIMENT_VALUE

    temp_data['Vader Sentiment'] = SENTIMENT

    return temp_data
