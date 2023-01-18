# import spacy module
import spacy
# load the advanced language model
nlp = spacy.load('en_core_web_md')

# ======== Example 1 =============

# determine the similarity between the following words
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

# use similarity command from spacy to return similarity rating 0 to 1
print(word1, word2, word1.similarity(word2))
print(word3, word2, word3.similarity(word2))
print(word3, word1, word3.similarity(word1))

'''
OUTPUT:
cat monkey 0.5929930274321619
banana monkey 0.40415016164997786
banana cat 0.22358825939615987

NOTE:
I note that the similarity between cat and monkey is highest as they are both animals.
The relationship between banana and monkey is higher than banana and cat, which makes sense because
monkeys eat bananas and the two are generally associated with each other. I actually would have guessed
the similarity between monkey and banana would be the highest, even though monkey and cat come out on top.
'''

# EXAMPLE OF MY OWN
word1 = nlp("london")
word2 = nlp("rain")
word3 = nlp("umbrella")
print(word1, word2, word1.similarity(word2))
print(word3, word2, word3.similarity(word2))
print(word3, word1, word3.similarity(word1))

'''
OUTPUT:
london rain 0.04984164607151575
umbrella rain 0.18561178798680752
umbrella london 0.10061059333774054

NOTE:
I was surprised with the relationship scores coming out so low.
As expected the highest relationship was with umbrella and rain.
'''

# ======== Example 2 =============

# compare a series of words with one another
tokens = nlp('cat apple monkey banana')

# use the nested for loop to compare each word in the series
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

'''
OUTPUT:
cat cat 1.0
cat apple 0.2036806046962738
cat monkey 0.5929930210113525
cat banana 0.2235882580280304
apple cat 0.2036806046962738
apple apple 1.0
apple monkey 0.2342509925365448
apple banana 0.6646699905395508
monkey cat 0.5929930210113525
monkey apple 0.2342509925365448
monkey monkey 1.0
monkey banana 0.4041501581668854
banana cat 0.2235882580280304
banana apple 0.6646699905395508
banana monkey 0.4041501581668854
banana banana 1.0
'''

# ======== Example 3 =============

# compare the similarity of the below sentence with the following sentences
sentence_to_compare = "Why is my cat on the car"

sentences = ["Where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

# iterate through each sentence in the list to compare each sentence with the target sentence
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

'''
OUTPUT:
Where did my dog go -  0.6085941301496852
Hello, there is my car -  0.8033180111627156
I've lost my car in my car -  0.6787541571030323
I'd like my boat back -  0.5624940517078084
I will name my dog Diana -  0.6491444739190607
'''

'''
NOTE ON USING THE SIMPLER LANGUAGE MODEL:

When running the example.py file with a simpler language model we received the following warning message:

UserWarning: [W007] The model you're using has no word vectors loaded,
so the result of the Doc.similarity method will be based on the tagger,
parser and NER, which may not give useful similarity judgements.
This may happen if you're using one of the small models,
e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors.
You can always add your own word vectors, or use one of the larger models instead if available.

<<<The output may not be useful using this language model!>>>
'''