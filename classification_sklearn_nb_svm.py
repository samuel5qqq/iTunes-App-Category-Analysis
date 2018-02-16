import pandas as pd
import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import itertools
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy as np
import random

##
# read data from a csv file
# @param    filename    csv file name
# @return   app_name    list of application name
#           description list of description
#           category    list of category
def read_genre_csv(filename):
    app_name = []
    description = []
    category = []

    with open(filename, "r",encoding='utf-8', errors='ignore') as csvDataFile:

        csvReader = csv.reader(csvDataFile)
        for row in csvReader:

            app_name.append(row[0])
            description.append(row[1])
            category.append(category_dict[row[4]])

    return app_name, description, category


##
# class that create a count vectorizer and reduce unnecessary words with stemmer
# @param    CountVectorizer
# @return   stemmed CountVectorizer
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


##
# print the accuracy in percentage
# @param    num     accuracy of the classification
def print_accuracy(num):
    print(str('{0:.2f}'.format(num * 100)) + '%')


##
# data analysis using Confusion Matrix
# @param    test        list of category in test set
# @param    predicted   list of category in predicted set
def data_analysis(test, predicted):
    matrix = metrics.confusion_matrix(test, predicted)
    #print('Confusion matrix:')
    #print(matrix)

    #for prediction, label in zip(predicted, test):
    #    if prediction != label:
    #        print('Classified as:', reversed_dict[prediction], ',\tand should be', reversed_dict[label])
    return matrix

##
# visualize data by plotting confusion matrix.
# normalization can be applied by setting 'normalize' to True
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nNormalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


##
# visiualize data
# @param    matrix      confusion matrix
# @param    class_name  each category's name
# @param    input_title title of the plot
def visiulize(matrix, class_name, input_title):
    plt.figure()
    plot_confusion_matrix(matrix, classes=class_name, normalize=True,
                          title=input_title)
    plt.show()

# dictionary for different categories of applications
category_dict = {'Business': 1, 'Education': 2, 'Sports': 3, 'Weather': 4, 'Music': 5, 'Games': 6, 'News': 7, 'Travel': 8, 'Photo & Video': 9, 'Shopping': 10}
reversed_dict = {1: 'Business', 2: 'Education', 3: 'Sports', 4: 'Weather', 5: 'Music', 6: 'Games', 7: 'News', 8: 'Travel', 9: 'Photo & Video', 10: 'Shopping'}
class_names = ['Business', 'Education', 'Sports', 'Weather', 'Music', 'Games', 'News', 'Travel', 'Photo & Video', 'Shopping']

# read csv files
app_name1, description1, category1 = read_genre_csv('ios-businessoutput.csv')
app_name2, description2, category2 = read_genre_csv('ios-educationoutput.csv')
app_name3, description3, category3 = read_genre_csv('ios-sportsoutput.csv')
app_name4, description4, category4 = read_genre_csv('ios-weatheroutput.csv')
app_name5, description5, category5 = read_genre_csv('ios-musicoutput.csv')
app_name6, description6, category6 = read_genre_csv('ios-gamesoutput.csv')
app_name7, description7, category7 = read_genre_csv('ios-newsoutput.csv')
app_name8, description8, category8 = read_genre_csv('ios-traveloutput.csv')
app_name9, description9, category9 = read_genre_csv('ios-photo-videooutput.csv')
app_name10, description10, category10 = read_genre_csv('ios-shoppingoutput.csv')

# append all data sets together
description = description1 + description2 + description3 + description4 + description5 + description6 + description7 + description8 + description9 + description10
category = category1 + category2 + category3 + category4 + category5 + category6 + category7 + category8 + category9 + category10

# shuffle all the descriptions, categories lists together
description_category = list(zip(description, category))
random.shuffle(description_category)
description, category = zip(*description_category)

# split the data into train, test 80% and 20% respectively
train_size = int(len(description)*0.7)
print('Size of the training data:')
print(train_size)
description_train = description[:train_size]
description_test = description[train_size:]
category_train = category[:train_size]
category_test = category[train_size:]

# Convert a collection of text documents to a matrix of token counts
# fit the training data to CountVectorizer()
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(description_train)

#Document-Term matrix. [n_samples, n_features].
#X_train_counts.shape

# To avoid giving more weight to longer documents than shorter documents, use frequency (TF - Term Frequencies)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#Document-Term matrix. [n_samples, n_features].
print('Document-Term matrix. [n_samples, n_features].')
print(X_train_tfidf.shape)


#Naive Bayes Classification
# using pipeline to do the same work as above
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
text_clf = text_clf.fit(description_train, category_train)
predicted = text_clf.predict(description_test)
print('\n\nNaive Bayes without using nltk stemmed stop words:')
print_accuracy(np.mean(predicted == category_test))
data_analysis(category_test, predicted)
mnb_matrix = data_analysis(category_test, predicted)

#visiulize(mnb_matrix, class_names, 'Normalized NB confusion matrix')

# SVM Classification
text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
text_clf_svm = text_clf_svm.fit(description_train, category_train)
predicted_svm = text_clf_svm.predict(description_test)
print('\n\nSVM without using nltk stemmed stop words:')
print_accuracy(np.mean(predicted_svm == category_test))
data_analysis(category_test, predicted_svm)
svm_matrix = data_analysis(category_test, predicted_svm)

#visiulize(svm_matrix, class_names, 'Normalized SVM confusion matrix')

print('\n------------------------------------------\n')


# Using NLTK stop words (using multinomial Naive Bayes)
stemmer = SnowballStemmer("english", ignore_stopwords=True)

stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),('tfidf', TfidfTransformer()),('mnb', MultinomialNB(fit_prior=False)),])
text_mnb_stemmed = text_mnb_stemmed.fit(description_train, category_train)
predicted_mnb_stemmed = text_mnb_stemmed.predict(description_test)
print('Naive Bayes using nltk stemmed stop words:')
print_accuracy(np.mean(predicted_mnb_stemmed == category_test))
mnb_stemmed_matrix = data_analysis(category_test, predicted_mnb_stemmed)
# plot the resulting data
visiulize(mnb_stemmed_matrix, class_names, 'Normalized stemmed NB confusion matrix')


# Using NLTK stop words (using SVM classification)
text_svm_stemmed = Pipeline([('vect', stemmed_count_vect),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
text_svm_stemmed = text_svm_stemmed.fit(description_train, category_train)
predicted_svm_stemmed = text_svm_stemmed.predict(description_test)
print('\n\nSVM using nltk stemmed stop words:')
print_accuracy(np.mean(predicted_svm_stemmed == category_test))
svm_stemmed_matrix = data_analysis(category_test, predicted_svm_stemmed)

# plot the resulting data
visiulize(svm_stemmed_matrix, class_names, 'Normalized stemmed SVM confusion matrix')