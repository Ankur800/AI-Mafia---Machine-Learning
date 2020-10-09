from nltk.corpus import brown

# 1.DATA COLLECTION
# print(brown.categories())

data = brown.sents(categories='editorial')[:100]
# print(type(data), len(data))
# print(data)

# ************** NLP PIPELINE *******************
# - data collection                             *
# - tokenization, stopwards removal, stemming   *
# - building a common vocab                     *
# - vectorize the documents                     *
# - performing classification/clustering        *
# ***********************************************

# 2. TOKENIZATION AND STOPWARD REMOVAL

text = "It was a very pleasant day, the weather was cool and there were showers. I went to market to buy some fruits."

from nltk.tokenize import sent_tokenize, word_tokenize

sents = sent_tokenize(text)
# print(sents)

word_list = word_tokenize(sents[0].lower())
# print(word_list)

# STOPWORDS REMOVAL
from nltk.corpus import stopwords

sw = set(stopwords.words('english'))


# print("The stopwords are : ")
# print(sw, len(sw))

# let's remove stopwords from our text
# FILTER THE WORDS FROM THE SENTENCE
def filter_words(word_list):
    useful_words = [w for w in word_list if w not in sw]
    return useful_words


useful_words = filter_words(word_list)
# print(useful_words)

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer("[a-zA-Z0-9]+")
sentence = "send the 50 documents to abc, def, ghi."
# print(tokenizer.tokenize(sentence))

# ******************** STEMMING *****************************
# -process that transforms particular words into roo words  *
# -jumping, jumps, jump, jumped -> jump                     *
# ***********************************************************

text = "The quick brown fox was seen jumping over the lazy dog from high wall. Foxes love to make jumps."

word_list = tokenizer.tokenize(text.lower())
# print(word_list)

# ****** TYPES OF STEMMERS **********
# -Snowball stemmer (Multilingual)  *
# -Porter stemmer                   *
# -Lancaster stemmer                *
# ***********************************

from nltk.stem.snowball import PorterStemmer, SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
ps = PorterStemmer()
print(ps.stem("crowded"))