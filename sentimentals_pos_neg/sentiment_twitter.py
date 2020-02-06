# - *- coding: utf- 8 - *-
from textblob import TextBlob
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s




#consumer key, consumer secret, access token, access secret.
ckey="KvVdfB4DtxvzxvsKhaAEJRlK4"
csecret="y6ys8nryuk38NhRGuJOiOT6nNJ4ChOfYvXl5SvkT1gW3yOxKEO"
atoken="944854539055886338-C0KshuXpjPmNAWprSN0Lxr1r0CLcR8c"
asecret="YNMovhC6vbztJMDdZ2B4Gc4gtCGqRLvfXEBLXe2ix4LgG"

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)

        tweet = all_data["text"]
        # check if language is english
        tweet_blob = TextBlob(tweet)
        lang = tweet_blob.detect_language()

        if lang == "en" and "RT" not in tweet and "https" not in tweet:
            sentiment_value, confidence = s.sentiment(tweet)
            print(tweet, sentiment_value, confidence)

            if confidence >= 0.8:
                output = open("twitter-out.txt", "a")
                output.write(sentiment_value)
                output.write("\n")
                output.close()

        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["1917"])