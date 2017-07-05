#This lda is based on gensim
#source is 
#https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/topic_methods.ipynb

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
#pip3 install stop-words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pyLDAvis
import pyLDAvis.gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# create sample documents
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health." 

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)


#print(texts)
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
hdpmodel = gensim. models.hdpmodel.HdpModel(corpus, id2word = dictionary)

print("----------------------------")
print("topics are")
print(hdpmodel.show_topics(num_topics=3, num_words=3))
print("----------------------------")


print("----------------------------")
print(texts[0])
print(corpus[0])
ldamodel = hdpmodel.suggested_lda_model()
doc_topics, word_topics, phi_values = ldamodel.get_document_topics(corpus[0], per_word_topics=True)
print(doc_topics)#gives topic distribution of the document[0]
print(word_topics)#gives topic distribution for each word in the document[0].
print("----------------------------")


data =  pyLDAvis.gensim.prepare(hdpmodel, corpus, dictionary)
pyLDAvis.display(data)
pyLDAvis.save_html(data, 'test_lda.html')


'''
outputs are
----------------------------
topics are
[(0, '0.074*"brother" + 0.074*"mother" + 0.074*"drive"'), (1, '0.094*"brocolli" + 0.094*"good" + 0.094*"health"'), (2, '0.031*"drive" + 0.031*"pressur" + 0.031*"brother"')]
----------------------------
----------------------------
 returns the odds of that particular word belonging to a particular topic.
[(1, 0.081060987847040217)]
----------------------------
----------------------------
['brocolli', 'good', 'eat', 'brother', 'like', 'eat', 'good', 'brocolli', 'mother']
[(0, 2), (1, 2), (2, 1), (3, 1), (4, 1), (5, 2)]
[(0, 0.036712546884993959), (1, 0.92948849543648004), (2, 0.033798957678526018)]
[(0, [1]), (1, [1]), (2, [1, 0]), (3, [1, 0]), (4, [1]), (5, [1])]
----------------------------
'''