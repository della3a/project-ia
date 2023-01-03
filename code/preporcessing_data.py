# Import the necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from data_reader import read_data

vectorizer = CountVectorizer()


def preprocess_data(tweets_df):

    # shape of the data
    #print(tweets_df.shape)

    # drop the columns that are not needed
    tweets_df = tweets_df.drop('Index', axis=1)

    # rename the columns to make it easier to understand
    tweets_df.rename(columns={'message to examine': 'message',
                              'label (depression result)': 'label'}, inplace=True)

    # clean the data
    tweets_df = tweets_df.reset_index()

    # Clean and preprocess the data
    # Convert the text to lowercase
    tweets_df['message'] = tweets_df['message'].str.lower()

    tweets_df['message'] = tweets_df['message'].str.replace(
        r'[^\w\s]', '', regex=False)  # Remove punctuation
    tweets_df['message'] = tweets_df['message'].str.replace(
        r'\d', '', regex=False)  # Remove digits

    # Remove leading and trailing whitespace
    tweets_df['message'] = tweets_df['message'].str.strip()

    # Extract the text of the tweets and the labels
    X = tweets_df['message'].values  # The text of the tweets
    # The labels (assuming that 0 is negative, 1 is positive)
    y = tweets_df['label'].values

    # Create a CountVectorizer object to convert the tweets into a bag of words representation
    X = vectorizer.fit_transform(X)  # X is now a matrix of numerical features

    return X, y


preprocess_data(read_data(
    '/Users/manelkfc/Desktop/M1/techniques-d-apprentissage-artificiel/projet-techniques-apprentissage-artificiel/data/sentiment_tweets.csv'))


def preprocess_tweet(tweet):
    # Convert the tweet to lowercase
    tweet = tweet.lower()
    # Remove punctuation
    tweet = tweet.replace(r'[^\w\s]', '')
    # Remove digits
    tweet = tweet.replace(r'\d', '')
    # Remove leading and trailing whitespace
    tweet = tweet.strip()

    # Convert the tweet to a numerical representation (e.g. using a CountVectorizer)
    tweet = vectorizer.transform([tweet])

    return tweet
