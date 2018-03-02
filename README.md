# Project descrption 

Implemented multiple classification training models to predict the category of a random testing application. Used description of each application in the training set to form the training model.

Classification Algorithm (1)Multinomial Naïve Bayes (2)Support Vector Machine (3)K-NN (4)Random Forest (5)Multiple Layer Perceptron

# Program Process

### Scrap data link from iTunes store
- First, use beautiful soup to scarp link from iTunes store based on different categories. 
- For each category, we scrapped every app link that belongs to it. Saved every category’s link to csv file.

### Scrap app information from itune store
- Here scrapped the category’s app information for data analysis. Used the app link that scrapped previously from app_crawler.py. 
- For analysis, scrapped 10 categories: [games, education, business, music, sports, weather, photo-video, shopping, news, travel] Scrapped each category for 200 app data.

### Preprocessing Data
- Build a multiclassification model with 10 output categories: Business, Education, Sports, Weather, Music, Games, News, Travel, Photo & Video, and Shopping. Loaded 10 categories’ csv files into the program, extracting description of each application along with its category. 
- Sampled 200 applications for each category, so we have 2000 sampled data. Shuffled all description data list and categorized data list together, and divided the data into training set and testing set (70% training).

### Vectorize the text documents
- Using CountVectorizer from sklearn.feature_extraction library to convert a collection of text documents to a matrix of token counts. Document-Term matrix is a useful tool to see how many samples and features we get in the training set. After vectorizing the text documents, obtained 20686 word features.

### Reducing unnecessary words
- Used TfidfTransformer() from sklearn.feature_extraction to change word counts to term frequencies. It avoids giving more weight to longer documents than shorter documents. Stop words (this, the etc.) from the data are usually not useful for classification. 
- Use SnowballStemmer from nltk.stem.snowball to remove stop words and reduce words to root form (e.g. from “fishing”, “fished”, to “fish”). The result of removing stop words increased accuracy by ~ 1%.

### Data Visualization: confusion matrix
- Used matrics.confusion_matrix from sklearn to present the result in matrix form. Then used matplotlib.pyplot to plot out the confusion matrix. To make the data easier to interpret, normalized the data so each row of accuracy weight summed to 1.

# Running the Program

1. Scrap app link from iTunes Store <br/>
python app_crawler.py
2. Scrap app information. Argument 1 is the number of app you want to scrap.<br/>
python app_info_crawler.py <number>
3. K-NN and random forest method<br/>
python classification_sklearn_knn_rf.py
4. Multiple Layer Perceptron<br/>
python classification_sklearn_mlp.py
5. Multinomial Naïve Bayes and Support Vector Machine<br/>
  python classification_sklearn_nb_svm.py
