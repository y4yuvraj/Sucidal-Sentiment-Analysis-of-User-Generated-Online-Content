#%%
#importing necessary libraries
import nltk
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt 
import warnings
from nltk.stem import WordNetLemmatizer #word stemmer class
lemma = WordNetLemmatizer()
DeprecationWarning('ignore')
os.chdir('C:/Users/_yuv_/Desktop/mac_learn/mini')
warnings.filterwarnings('ignore')
# %%
#importing dataset
train=pd.read_excel('data.xlsx')
#feature engineering starts
#%%
#ananlyzing data
train.info()
train.head()
#%%
#droping rows with no text
train.dropna(inplace=True)
# %%
#counting negative and positive sentiments
print(len(train[train.sentiment == -1]), 'depressing Tweets')
print(len(train[train.sentiment == 1]), 'non-depressing Tweets')
#%%
#changing some abbreviation
train.text.replace(to_replace="rn",value="right now")
train.text.replace(to_replace="ppl",value="people")
train.text.replace(to_replace="ur",value="your")
train.text.replace(to_replace="i'm",value="i am")

#%%
#removing special characters
spec_chars = ["â","€","œ","™","¦","ì","–","",">","ð","ÿ","˜","¤","¼","ï","¾","¹","à","ˆ","²","½","š","ø","ù","š","ƒ","ž","µ"]
for char in spec_chars:
    train['text'] = train['text'].str.replace(char, '')

#%%
#removing user name and retweet and also removing hyperlinks
import re
def normalizer(text):
    tweets = " ".join(filter(lambda x: x[0]!= '@' , text.split()))
    tweets = re.sub("(@[A-Za-z0-9_]+)","", tweets)
    tweets = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+',' ', tweets)
    tweets=re.sub('RT','',tweets)
    tweets = tweets.lower()
    tweets = [lemma.lemmatize(word) for word in tweets]
    tweets = "".join(tweets)
    return tweets
train['normalized_text'] = train.text.apply(normalizer)
   
#%%
#Extracting words with hashtag for further analysis
def extract_hashtag(tweet):
    tweets = " ".join(filter(lambda x: x[0]== '#', tweet.split()))
    tweets = re.sub('[^a-zA-Z]',' ',  tweets)
    tweets = tweets.lower()
    tweets = [lemma.lemmatize(word) for word in tweets]
    tweets = "".join(tweets)
    return tweets

train['hashtag'] = train.normalized_text.apply(extract_hashtag)

#%%
#remove common words
from nltk.corpus import stopwords
set(stopwords.words('english'))
stop = stopwords.words('english')

def stop_words_removal(df):
    df['n_text_sw'] = df['normalized_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    print(df['n_text_sw'].head())
stop_words_removal(train)
#%%
#PUNCTUATION REMOVAL
def punctuation_removal(df):
    df['text'] = df['text'].str.replace('[^\w\s]','')
    df['n_text_sw'] = df['n_text_sw'].str.replace('[^\w\s]','')
    print(df['text'].head())

# %%
punctuation_removal(train)
#%%
#checking 10 most frequently occuring words in data
freq = pd.Series(' '.join(train['n_text_sw']).split()).value_counts()[4:10]
freq
# %%
#removing these words as their presence will not of any use in classification of our text data.
freq = list(freq.index)
#%%
def frequent_words_removal(df):    
    df['n_text_sw'] = df['n_text_sw'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    print(df['n_text_sw'].head())
# %%
frequent_words_removal(train)
#%%
#checking rare occuring words
freq = pd.Series(' '.join(train['n_text_sw']).split()).value_counts()[-19:]
freq
# %%
#removing rare words
freq = list(freq.index)
def rare_words_removal(df):
    df['n_text_sw'] = df['n_text_sw'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    print(df['n_text_sw'].head())

#%%
#To Create Cloud of words for all words and hatred words
# all tweets 
all_words = " ".join(train.n_text_sw)

# %%
#Hatred tweets
hatred_words = " ".join(train[train['sentiment']==-1].n_text_sw)
#print(hatred_words)

# %%
from wordcloud import WordCloud, STOPWORDS

# %%
#All tweets cloudword
wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white')
wordcloud = wordcloud.generate(all_words)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#%%
#hatred tweets cloudword
wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white')
wordcloud = wordcloud.generate(hatred_words)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# %%
#Analysing Hashtag words
from nltk import FreqDist 
freq_all_hashtag = FreqDist(list((" ".join(train.hashtag)).split())).most_common(10)
freq_all_hashtag

# %%
#hatred hastag
freq_hatred_hashtag = FreqDist(list((" ".join(train[train['sentiment']==-1]['hashtag'])).split())).most_common(10)
freq_hatred_hashtag

# %%
df_allhashtag = pd.DataFrame(freq_all_hashtag, columns=['words', 'frequency'])
df_hatredhashtag = pd.DataFrame(freq_hatred_hashtag, columns=['words', 'frequency'])
print(df_allhashtag.head())

# %%
#visualising all hashtags
import seaborn as sns
sns.barplot(x='words', y='frequency', data=df_allhashtag)
plt.xticks(rotation = 45)
plt.title('hashtag words frequency')
plt.show()

# %%
#hatred hashtog
sns.barplot(x='words', y='frequency', data=df_hatredhashtag)
plt.xticks(rotation = 45)
plt.title('hatred hashtag frequency')
plt.show()
#%%
#Visualizing the classes of train data
chat_data = train['sentiment'].value_counts()
plt.pie(chat_data, autopct='%1.1f%%', shadow=True,labels=['Positive Class','Negative Class'])
plt.title('Class Distribution');
plt.show()

# %%
#Creating the length column for tweet
train['pre_clean_len']=  [len(t) for t in train.n_text_sw]
#Box plot of all data
fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(train.pre_clean_len)
plt.title('Word length of all tweets ')
plt.show()

# %%
#Let's look at exact numbers of positive and negative tweet length
print('\033[5m'+'Positive Tweets:'+"\033[0;0m")
print('Maximum number of words are',train[train['sentiment']==1].pre_clean_len.max())
print(' ')
print('\033[5m'+'Negative Tweets:'+"\033[0;0m")
print('Maximum number of words are',train[train['sentiment']==-1].pre_clean_len.max())

# %%
#Defining x and y
X = train['n_text_sw']
y = train['sentiment']
#%%
#Importing TFIDF 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tfidf = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 4), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
#%%
# to create sparse matrix corpus is created to pass to vectorizer
len(train)
corpus = []
for i in range(0,2076):
    corpus.append(train['n_text_sw'][i])
#corpus
#%%
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words=stopwords.words('english'))
cv.fit(corpus)
# %%
#Fitting TFIDF to both training
x_train_tfidf =  tfidf.fit_transform(corpus) 

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
import time
start_time = time.time()
param_grid = {'C': np.arange(20,30,2),
              'max_iter': np.arange(100,1200,100),
              'penalty': ['l1','l2']}

i=1
kf = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
for train_index,test_index in kf.split(x_train_tfidf,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl = x_train_tfidf[train_index],x_train_tfidf[test_index]
    ytr,yvl = y[train_index],y[test_index]
    
    model = RandomizedSearchCV(estimator=LogisticRegression(class_weight='balanced'),param_distributions=param_grid,verbose=0)
    

    model.fit(xtr, ytr)
    #print (model.best_params_)
    pred=model.predict(xvl)
    print('roc_auc_score',roc_auc_score(yvl,pred))
    i+=1

print("Execution time: " + str((time.time() - start_time)) + ' ms')
print ('best parameters',model.best_params_)
#%%
roc_auc_logistic = roc_auc_score(yvl,pred).mean()
f1_logistic = f1_score(yvl,pred).mean()
print('Mean - ROC AUC', roc_auc_logistic)
print('F1 Score - ', f1_logistic)
print('Confusion Matrix \n',confusion_matrix(yvl,pred))

# %%
#DecisionTree with tuned hyperparameters
from sklearn.tree import DecisionTreeClassifier
start_time = time.time()
param_grid = {'criterion': ['gini','entropy'],
             'min_samples_split':[50,70,100,150],
             'max_features': ['sqrt','log2']}


i=1
kf = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
for train_index,test_index in kf.split(x_train_tfidf,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl = x_train_tfidf[train_index],x_train_tfidf[test_index]
    ytr,yvl = y[train_index],y[test_index]
    
    model = RandomizedSearchCV(estimator=DecisionTreeClassifier(class_weight={1:1,1:5}),param_distributions=param_grid,verbose=0)
    

    model.fit(xtr, ytr)
    #print (model.best_params_)
    pred=model.predict(xvl)
    print('roc_auc_score',roc_auc_score(yvl,pred))
    i+=1

print("Execution time: " + str((time.time() - start_time)) + ' ms')
print ('best parameters',model.best_params_)

# %%
#Model Accuracy
roc_auc_dt = roc_auc_score(yvl,pred).mean()
f1_dt = f1_score(yvl,pred).mean()
print('Mean - ROC AUC', roc_auc_dt)
print('F1 Score - ', f1_dt)
print('Confusion Matrix \n',confusion_matrix(yvl,pred))
#%%
from sklearn.ensemble import RandomForestClassifier
start_time = time.time()
param_grid = {'criterion': ['entropy'],
             'min_samples_split':np.arange(10,100,20),
             'max_features': ['sqrt'],
             'n_estimators':[10,20,30]}

i=1
kf = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
for train_index,test_index in kf.split(x_train_tfidf,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl = x_train_tfidf[train_index],x_train_tfidf[test_index]
    ytr,yvl = y[train_index],y[test_index]
    
    model = RandomizedSearchCV(estimator=RandomForestClassifier(),param_distributions=param_grid,verbose=0)
    

    model.fit(xtr, ytr)
    #print (model.best_params_)
    pred=model.predict(xvl)
    print('roc_auc_score',roc_auc_score(yvl,pred))
    i+=1

print("Execution time: " + str((time.time() - start_time)) + ' ms')
print ('best parameters',model.best_params_)

# %%
#Model Accuracy
roc_auc_rf = roc_auc_score(yvl,pred).mean()
f1_rf = f1_score(yvl,pred).mean()
print('Mean - ROC AUC', roc_auc_rf)
print('F1 Score - ', f1_rf)
print('Confusion Matrix \n',confusion_matrix(yvl,pred))
#%%
#naive bayes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train['n_text_sw'], train['sentiment'], test_size=0.5, stratify=train['sentiment'])

# %%
trainp=train[train.sentiment==1]
trainn=train[train.sentiment==-1]
print(trainp.info())
trainn.info()

# %%
# Let us balance the dataset
train_imbalanced = train
from sklearn.utils import resample
df_majority = train[train.sentiment==1]
df_minority = train[train.sentiment==-1]
#%%
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=123) # reproducible results
#%% 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
#%% 
# Display new class counts
print("Before")
print(train.sentiment.value_counts())
print("After")
print(df_upsampled.sentiment.value_counts())

X_train, X_test, y_train, y_test = train_test_split(df_upsampled['n_text_sw'], df_upsampled['sentiment'], test_size=0.5, stratify=df_upsampled['sentiment'])

# %%
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

# %%
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vect = CountVectorizer()
tf_train=vect.fit_transform(X_train)  #train the vectorizer, build the vocablury
tf_test=vect.transform(X_test)  #get same encodings on test data as of vocabulary built

# %%
model.fit(X=tf_train,y=y_train)

# %%
expected = y_test
predicted=model.predict(tf_test)

# %%
from sklearn import metrics

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
#%%
f1_nb=(f1_score(y_test, predicted))
roc_auc_nb=roc_auc_score(y_test,predicted)
#%%
#Summary table for all models

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest','Naive Bayes'],
    'Mean - ROC AUC Score (Fold=10)': [roc_auc_logistic, roc_auc_dt, roc_auc_rf,roc_auc_nb],
    'Mean - F1 Score': [f1_logistic,f1_dt,f1_rf,f1_nb]})
results

# %%
