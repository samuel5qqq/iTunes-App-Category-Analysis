Project descrption: In the project, we implemented multiple classification training models to predict the category of a random testing application. We used description of each application in the training set to form the training model.

Classification Algorithm (1)Multinomial Naïve Bayes (2)Support Vector Machine (3)K-NN (4)Random Forest (5)Multiple Layer Perceptron

Scrap data link from iTunes store

First, use beautiful soup to scarp link from iTunes store based on different categories. For each category, we scrapped every app link that belongs to it. Saved every category’s link to csv file.

Scrap app information from itune store

Here we scrapped the category’s app information that we wanted to use for data analysis. Used the app link that we scrapped previously from app_crawler.py. For our analysis, we scrapped 10 categories: [games, education, business, music, sports, weather, photo-video, shopping, news, travel] Scrapped each category for 200 app data.

Preprocessing Data

We decided to build a multiclassification model with 10 output categories: Business, Education, Sports, Weather, Music, Games, News, Travel, Photo & Video, and Shopping. Loaded 10 categories’ csv files into the program, extracting description of each application along with its category. We sampled 200 applications for each category, so we have 2000 sampled data. Shuffled all description data list and categorized data list together, and divided the data into training set and testing set (70% training).

Vectorize the text documents

Using CountVectorizer from sklearn.feature_extraction library to convert a collection of text documents to a matrix of token counts. Document-Term matrix is a useful tool to see how many samples and features we get in the training set. After vectorizing the text documents, we obtained 20686 word features.

Reducing unnecessary words

Used TfidfTransformer() from sklearn.feature_extraction to change word counts to term frequencies. It avoids giving more weight to longer documents than shorter documents. Stop words (this, the etc.) from the data are usually not useful for classification. We use SnowballStemmer from nltk.stem.snowball to remove stop words and reduce words to root form (e.g. from “fishing”, “fished”, to “fish”). The result of removing stop words increased accuracy by ~ 1%.

Data Visualization: confusion matrix

We used matrics.confusion_matrix from sklearn to present the result in matrix form. Then used matplotlib.pyplot to plot out the confusion matrix. To make the data easier to interpret, we normalized the data so each row of accuracy weight summed to 1.
