from preporcessing_data import preprocess_tweet


def predict(new_tweet, clf):
    # Preprocess the new tweet
    # new_tweet = "I feel so depressed and hopeless today."
    tweet_data = preprocess_tweet(new_tweet)

    # Use the classifier's predict method to predict the label of the new tweet
    prediction = clf.predict(tweet_data.reshape(1, -1))

    return prediction


def classify_tweets(tweets, clf):
    # Keep track of the number of tweets classified as depressive
    num_depressive_tweets = 0

    # Classify each tweet
    for tweet in tweets:
        prediction = predict(tweet, clf)

        # Increment the count if the tweet is classified as depressive
        if prediction == 1:
            num_depressive_tweets += 1
            print("The tweet is classified as depressive.")
        elif prediction == 0:
            print("The tweet is classified as not depressive.")
            num_depressive_tweets += 0

    # Determine if more than half of the tweets are classified as depressive
    if num_depressive_tweets > len(tweets) / 2:
        print("This person is classified as having depressive thoughts.")
    else:
        print("This person is classified as not having depressive thoughts.")
