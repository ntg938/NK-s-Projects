## Sentiment Analysis for Movie Reviews

The attached code looks at using a Recurrent Neural Network in Keras to classify a movie review as either positive or negetive.

# Getting the dataset
We will be using the IMDb movie reviews dataset that conviniently for us Keras has a built in data set for us to use. The dataset contains 50,000 reviews with a even number of positive and negetive reviews. A positive review is one with a score of =>7 out of 10 and negetive is anything =<4 out of 10. Neutral reviews are not included in the dataset.

to load the data we will import as follow:

from keras.datasets import imdb
