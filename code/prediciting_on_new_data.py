from preporcessing_data import preprocess_tweet


def predict(new_tweet, clf):
    # Preprocess the new tweet
    # new_tweet = "I feel so depressed and hopeless today."
    tweet_data = preprocess_tweet(new_tweet)

    # Use the classifier's predict method to predict the label of the new tweet
    prediction = clf.predict(tweet_data.reshape(1, -1))

    # Print the prediction
    if prediction == 0:
        print("The tweet is classified as not depressive.")
    elif prediction == 1:
        print("The tweet is classified as depressive.")
