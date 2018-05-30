# Import the required libraries.
import tweepy
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
import itertools as it
import json
import logging
import os
import re
import string
import sys

from nltk.util import everygrams
from nltk.tokenize.casual import TweetTokenizer

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import nltk
from sklearn.cluster import KMeans


stopwords = stopwords.words('english')
english_vocab = set(w.lower() for w in nltk.corpus.words.words())    

dfcities = pd.read_csv('C:\SrinivasaSolutions_Projects\Tutorials\Projects\Python\data\india-cities-states.csv', usecols=['City'])

def get_city_name(tweet_city):
    city_name=""
    for index, row in dfcities.iterrows():
        if (tweet_city.lower().find(row["City"].lower()) != -1):
            city_name = row["City"]
            break
        if (city_name==""):
            city_name = tweet_city.lower()
    return city_name


def replace_urls(in_string, replacement=None):
    """Replace URLs in strings. See also: ``bit.ly/PyURLre``

    Args:
        in_string (str): string to filter
        replacement (str or None): replacment text. defaults to '<-URL->'

    Returns:
        str
    """
    replacement = '<-URL->' if replacement is None else replacement
    pattern = re.compile('(https?://)?(\w*[.]\w+)+([/?=&]+\w+)*')
    return re.sub(pattern, replacement, in_string)

def get_candidate(row):
    candidates = []
    text = row["text"].lower()
    if "clinton" in text or "hillary" in text:
        candidates.append("clinton")
    if "trump" in text or "donald" in text:
        candidates.append("trump")
    if "sanders" in text or "bernie" in text:
        candidates.append("sanders")
    return ",".join(candidates)

#tweets["candidate"] = tweets.apply(get_candidate,axis=1)


def process_tweet_text(tweet):
   if tweet.startswith('@null'):
       return "[Tweet not available]"
   tweet = re.sub(r'\$\w*','',tweet) # Remove tickers
   tweet = re.sub(r'https?:\/\/.*\/\w*','',tweet) # Remove hyperlinks
   tweet = re.sub(r'['+string.punctuation+']+', ' ',tweet) # Remove puncutations like 's
   twtok = TweetTokenizer(strip_handles=True, reduce_len=True)
   tokens = twtok.tokenize(tweet)
   tokens = [i.lower() for i in tokens if i not in stopwords and len(i) > 2 and  
                                             i in english_vocab]
   return tokens


def toDataFrame(tweets):

    DataSet = pd.DataFrame()

    DataSet['tweetID'] = [tweet.id for tweet in tweets]
    DataSet['tweetText'] = [tweet.text for tweet in tweets]
    DataSet['tweetRetweetCt'] = [tweet.retweet_count for tweet in tweets]
    DataSet['tweetFavoriteCt'] = [tweet.favorite_count for tweet in tweets]
    DataSet['tweetSource'] = [tweet.source for tweet in tweets]
    DataSet['tweetCreated'] = [tweet.created_at for tweet in tweets]
    DataSet['userName'] = [tweet.user.name for tweet in tweets]
    DataSet['userCreateDt'] = [tweet.user.created_at for tweet in tweets]
    DataSet['userDesc'] = [tweet.user.description for tweet in tweets]
    DataSet['userFollowerCt'] = [tweet.user.followers_count for tweet in tweets]
    DataSet['userFriendsCt'] = [tweet.user.friends_count for tweet in tweets]
    DataSet['userLocation'] = [tweet.user.location for tweet in tweets]
    # DataSet['userIndiaCity'] = "" 
    # DataSet['geoLatitude'] = [tweet.geo.coordinates[0] for tweet in tweets]
    # DataSet['geoLongitude'] = [tweet.geo.coordinates[1] for tweet in tweets]



    return DataSet


def main():
    # Make the graphs prettier
    # pd.set_option('display.mpl_style', 'default')

    consumerKey = 'XEhVy8Oxa4XOIjITEOT3ulMCn'
    consumerSecret = '9KVAzBPetcApgjbyU2SXTzOGil9VxQqF24lXxv9SHPzg1gGx7e'

    #Use tweepy.OAuthHandler to create an authentication using the given key and secret
    auth = tweepy.OAuthHandler(consumer_key=consumerKey, consumer_secret=consumerSecret)

    #Connect to the Twitter API using the authentication
    api = tweepy.API(auth)

    #Perform a basic search query where we search for the '#NipahVirus' in the tweets
    result = api.search(q='%NipahVirus') #%23 is used to specify '#'

    # Print the number of items returned by the search query to verify our query ran. Its 15 by default
   
    print("Tweets Count: " , len(result))

    tweet = result[0] #Get the first tweet in the result
    '''
    # Analyze the data in one tweet to see what we require
    for param in dir(tweet):
    #The key names beginning with an '_' are hidden ones and usually  not required, so we'll skip them
        if not param.startswith("_"):
            print ("%s : %s\n" % (param, eval('tweet.'+param))    )
    '''
    results = []

    #Get the first 5000 items based on the search query
    for tweet in tweepy.Cursor(api.search, q='%NipahVirus').items(5000):
        results.append(tweet)

    # Verify the number of items returned
    print (len(results))

    df = toDataFrame(results)
    #list_places = dfcities.loc[:,'City'].tolist()
      
    df['userIndiaCity'] =  df['userLocation'].apply(get_city_name)
    df['userUsedWord'] = df['tweetText'].str.contains("bat")

    '''
    for i, row in df.iterrows():
        #print(row['userLocation'])
        row['userIndiaCity'] = df.apply(lambda row: get_city_name(row['userLocation']), axis=1)
       
        #row['userIndiaCity'] = get_city_name(row['userLocation'])
    '''
    # get_city_name("Ahmadabad City, India")
    # Let's check the top 5 records in the Data Set
    print(df.head(5))

    tweets_data = df["tweetText"].tolist()
   
    words = []
    for tw in tweets_data:
        words += process_tweet_text(tw)
    
    # print(words)

    data_to_vectorize = words

    from sklearn.feature_extraction.text import CountVectorizer
 
    # create the transform
    vectorizerC = CountVectorizer()
    # tokenize and build vocab
    vectorizerC.fit(data_to_vectorize)
    # summarize
    # print(vectorizerC.vocabulary_)
    # encode document
    vectorC = vectorizerC.transform(data_to_vectorize)
    # summarize encoded vector
    # print(vectorC.shape)
    # print(type(vectorC))
    # print(vectorC.toarray())
     
    from sklearn.feature_extraction.text import HashingVectorizer
  
    # create the transform
    vectorizerH = HashingVectorizer(n_features=20)
    # encode document
    vectorH = vectorizerH.transform(data_to_vectorize)
    # summarize encoded vector
    # print(vectorH.shape)
    # print(vectorH.toarray())    



    from sklearn.feature_extraction.text import TfidfVectorizer
   
    # create the transform
    vectorizerT = TfidfVectorizer()
    # tokenize and build vocab
    vectorizerT.fit(data_to_vectorize)
    # summarize
    # print(vectorizerT.vocabulary_)
    # print(vectorizerT.idf_)
    # encode document
    vectorT = vectorizerT.transform([data_to_vectorize[0]])
    
    
  
    # summarize encoded vector
    # print(vectorT.shape)
    # print(vectorT.toarray())
    
    X = vectorizerC.fit_transform(data_to_vectorize)
    terms = vectorizerC.get_feature_names()


    true_k = 2
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)

    print(X[0:10,:])

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    for i in range(true_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
        print

    print("\n")
    print("Prediction")        
 

    
# Calling main function
if __name__=="__main__":
    main()    