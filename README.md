## Sentiment Analysis for Movie Reviews

The attached code looks at using a Recurrent Neural Network in Keras to classify a movie review as either positive or negetive.

# Getting the dataset
We will be using the IMDb movie reviews dataset that conviniently for us Keras has a built in data set for us to use. The dataset contains 50,000 reviews with a even number of positive and negetive reviews. A positive review is one with a score of =>7 out of 10 and negetive is anything =<4 out of 10. Neutral reviews are not included in the dataset.

to load the data we will import as follow:

from keras.datasets import imdb

We then use the load_data function to load the data.

After the data has been loaded you pick up that it has been preprocessed for us aleady. If you run the following print commands, the out shows integers have already been mapped to words.

in : print('---review---')
     print(X_train[6])
     print('---label---')
     print(y_train[6])
     
out: -review---
[1, 2, 365, 1234, 5, 1156, 354, 11, 14, 2, 2, 7, 1016, 2, 2, 356, 44, 4, 1349, 500, 746, 5, 200, 4, 4132, 11, 2, 2, 1117, 1831, 2, 5, 4831, 26, 6, 2, 4183, 17, 369, 37, 215, 1345, 143, 2, 5, 1838, 8, 1974, 15, 36, 119, 257, 85, 52, 486, 9, 6, 2, 2, 63, 271, 6, 196, 96, 949, 4121, 4, 2, 7, 4, 2212, 2436, 819, 63, 47, 77, 2, 180, 6, 227, 11, 94, 2494, 2, 13, 423, 4, 168, 7, 4, 22, 5, 89, 665, 71, 270, 56, 5, 13, 197, 12, 161, 2, 99, 76, 23, 2, 7, 419, 665, 40, 91, 85, 108, 7, 4, 2084, 5, 4773, 81, 55, 52, 1901]
---label---
1

The integers above are sorted by how frequent a word appears.So 4 represents the 4th most used word, 5 the 5th most used word and so on. The integer 1 is reserved for the start marker, the integer 2 for an unknown word and 0 for padding. Thelabel is also an integer (0 for negative, 1 for positive).

We begin by splitting our training and testing data, and setting the number of words or vocab size.

in: (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 5000)
print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))

out: Loaded dataset with 25000 training samples, 25000 test samples

To see the words mapped to these intergers you run the following commands.
in :word2id = imdb.get_word_index()
    id2word = {i: word for word, i in word2id.items()}
    print('---review with words---')
    print([id2word.get(i, ' ') for i in X_train[6]])
    print('---label---')
    print(y_train[6])
    
You should see the follwing output:

out: -review with words---
['the', 'and', 'full', 'involving', 'to', 'impressive', 'boring', 'this', 'as', 'and', 'and', 'br', 'villain', 'and', 'and', 'need', 'has', 'of', 'costumes', 'b', 'message', 'to', 'may', 'of', 'props', 'this', 'and', 'and', 'concept', 'issue', 'and', 'to', "god's", 'he', 'is', 'and', 'unfolds', 'movie', 'women', 'like', "isn't", 'surely', "i'm", 'and', 'to', 'toward', 'in', "here's", 'for', 'from', 'did', 'having', 'because', 'very', 'quality', 'it', 'is', 'and', 'and', 'really', 'book', 'is', 'both', 'too', 'worked', 'carl', 'of', 'and', 'br', 'of', 'reviewer', 'closer', 'figure', 'really', 'there', 'will', 'and', 'things', 'is', 'far', 'this', 'make', 'mistakes', 'and', 'was', "couldn't", 'of', 'few', 'br', 'of', 'you', 'to', "don't", 'female', 'than', 'place', 'she', 'to', 'was', 'between', 'that', 'nothing', 'and', 'movies', 'get', 'are', 'and', 'br', 'yes', 'female', 'just', 'its', 'because', 'many', 'br', 'of', 'overly', 'to', 'descent', 'people', 'time', 'very', 'bland']
---label---
1

#Preprocessing the data

For our RNN to be able get this data fed into it, needs to have same length of input documents. In this regard the maximum review length will be limited to max_words and we will pad the shorter reviews, with 0, using the pad_sequence() function.
