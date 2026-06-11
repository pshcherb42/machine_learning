# nltk library for naturale language processing
# texblob used for sentiment analysis
# newspaper3k a library with 3k newspaper articles

from textblob import TextBlob

with open('mytext.txt', 'r') as f:
    text = f.read()

# article url sentiment translation
# take out commas and comment the .txt part
'''
from newspaper import Article

# fist step is to get article into the script
url = 'https://edition.cnn.com/2026/06/11/politics/pulte-dni-trump-administration-acting-officials-analysis'
# transform url into article object of the newspaper library
# this is the object that we're going to use to get the article into the script
article = Article(url)

# get the clean summary of the text into the sript
# get the article into the scrypt
article.download()
# make it 'readable'
article.parse()
# prepare it for natural language processing
article.nlp()

# get the text of the article 
text = article.summary
print(text)
'''
# blob object
blob = TextBlob(text)
sentiment = blob.sentiment.polarity # -1 to 1
print(sentiment)

