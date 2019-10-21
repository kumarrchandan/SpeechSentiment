from nltk.sentiment.vader import SentimentIntensityAnalyzer
#import analyzer from nltk


def vader_analyze(text_columns):
    all_sent_values = []
    #runs text column through Vader and returns the scores in a list in the same order as text
    for i in range(0,text_columns.size):
        all_sent_values.append(sentiment_value(str(text_columns[i])))
    return all_sent_values


def sentiment_value(paragraph):
   analyzer = SentimentIntensityAnalyzer()
   #the compound score is -1 to +1, -1 is most negative
   result = analyzer.polarity_scores(paragraph)
   score = result['compound']
   return round(score,2)