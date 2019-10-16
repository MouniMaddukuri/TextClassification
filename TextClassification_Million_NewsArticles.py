#!/usr/bin/env python
# coding: utf-8

# In[15]:


import json,os
import pprint as pp


# In[34]:


data_list=[]
sno=1
df = pd.DataFrame()
categories = ['Business','Entertainment','Politics','Sport','Tech']
directory = r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_BBC_Data\bbc-fulltext\bbc'
for cat in categories:
    dirname = os.path.join(directory,cat)
    for filename in os.listdir(dirname):
        file = os.path.join(dirname,filename)
        data = {}
        with open(file, 'r') as file:
            content = file.read().replace('\n', ' ')
            data['number'] = sno
            print(type(content))
            data['Content'] = content
            data['type']= cat
        df.append(data,ignore_index=True) 
        sno=sno+1
#json_data = json.dumps(data_list)
#df =  data['Content'](data_list)
#print(type(df['Content']))
#pp.pprint(json_data)


# In[35]:


print(df.head(1))


# In[17]:


from nltk.stem import WordNetLemmatizer
import nltk
import string as str
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('wordnet')
punctuation_signs=list('''!()-[]{};:'"\,<>./?#^_~''')
wordnet_lemmatizer = WordNetLemmatizer()
stop_words =list(stopwords.words('english')) + punctuation_signs
category_codes = {
    'Business': 0,
    'Entertainment': 1,
    'Politics': 2,
    'Sport': 3,
    'Tech': 4
}


# In[20]:


get_ipython().run_line_magic('time', "df3 = df['Content'].swifter.apply(process_text)")


# In[6]:


import json,os
import pandas as pd
import pprint as pp


# In[22]:


data_list=[]
sno=1
categories = ['Business','Entertainment','Politics','Sport','Tech']
directory = r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_BBC_Data\bbc-fulltext\bbc'
for cat in categories:
    dirname = os.path.join(directory,cat)
    for filename in os.listdir(dirname):
        file = os.path.join(dirname,filename)
        data = {}
        with open(file, 'r') as file:
            content = file.read().replace('\n', ' ')
            data['number'] = sno
            data['Content'] = content
            data['Category']= cat
        data_list.append(data) 
        sno=sno+1
json_data = json.dumps(data_list)
print(sno)
#pp.pprint(json_data)


# In[23]:


df_bbc = pd.DataFrame(data_list)
df_bbc.head(1)


# In[24]:


import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
import string as str
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('wordnet')
punctuation_signs=list('''!()-[]{};:'"\,<>./?#^_~''')
wordnet_lemmatizer = WordNetLemmatizer()
stop_words =list(stopwords.words('english')) + punctuation_signs
category_codes = {
    'Business': 0,
    'Entertainment': 1,
    'Politics': 2,
    'Sport': 3,
    'Tech': 4
}


# In[26]:


import numpy as np 
import concurrent.futures
def process_text(content,category):
        dict_parsed = {}
        res = [wordnet_lemmatizer.lemmatize(word, pos="v") for word in nltk.word_tokenize(content.lower().replace("'s"," ")) if word.lower() not in stop_words]
        #dict_parsed['id'] = pk
        dict_parsed['Content_Parsed'] = ' '.join(res)
        dict_parsed['Category_code'] = category
        dict_parsed['News_length'] = len(content)
        return dict_parsed 


# In[28]:


series_bbc = df_bbc[['Content', 'Category']].swifter.apply(lambda row: process_text(row['Content'], row['Category']), axis=1)


# In[31]:


df_process_bbc = series_bbc.swifter.apply(pd.Series)


# In[32]:


df_process_bbc.head(1)


# In[33]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(df_process_bbc['Content_Parsed'], 
                                                    df_process_bbc['Category_code'], 
                                                    test_size=0.15, 
                                                    random_state=8)


# In[35]:


from sklearn.feature_extraction.text import TfidfVectorizer
# Parameter election
ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 300


# In[36]:


tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                        
features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
print(features_train.shape)

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
print(features_test.shape)


# In[37]:


from sklearn.feature_selection import chi2
import numpy as np

for Product, category_id in sorted(category_codes.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' category:".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    print("")


# In[39]:


Export = df_process_bbc.to_json(r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_files\df_trainedSet.jsonl',orient='records', lines=True)
Export = X_train.to_json(r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_files\X_train.jsonl',orient='records', lines=True)
Export = y_train.to_json(r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_files\Y_train.jsonl',orient='records', lines=True)
Export = X_test.to_json(r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_files\X_test.jsonl',orient='records', lines=True)
Export = y_test.to_json(r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_files\Y_test.jsonl',orient='records', lines=True)


# In[40]:


dataframe = pd.DataFrame.from_records(features_train)
Export = dataframe.to_json(r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_files\features_train.jsonl',orient='records', lines=True)


# In[41]:


print(type(labels_train))
print(type(features_test))
print(type(labels_test))
print(type(tfidf))


# In[42]:


features_test_df = pd.DataFrame.from_records(features_test)
Export = labels_train.to_json(r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_files\labels_train.jsonl',orient='records', lines=True)
Export = features_test_df.to_json(r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_files\features_test.jsonl',orient='records', lines=True)
Export = labels_test.to_json(r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_files\labels_test.jsonl',orient='records', lines=True)


# In[43]:


idf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
idf = pd.DataFrame(columns=['idf']).from_dict(dict(idf), orient='index')
idf.columns = ['idf']


# In[44]:


#idf = dict(tfidf.get_feature_names(), tfidf.idf_)


# In[45]:


idf


# In[46]:


x = tfidf.get_feature_names()
print(len(x))


# In[47]:


Export = idf.to_json(r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_files\tfidf.jsonl',orient='records', lines=True)


# In[48]:


import pickle


# In[49]:


with open(r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_files\tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)


# # SVC Hyper parameter model prediction:

# In[50]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns


# In[51]:


svc_0 =svm.SVC(random_state=8)

print('Parameters currently in use:\n')
pprint(svc_0.get_params())


# In[52]:


print(features_train.shape)
print(features_test.shape)


# In[ ]:


# C
C = [.0001, .001, .01]

# gamma
gamma = [.0001, .001, .01, .1, 1, 10, 100]

# degree
degree = [1, 2, 3, 4, 5]

# kernel
kernel = ['linear', 'rbf', 'poly']

# probability
probability = [True]

# Create the random grid
random_grid = {'C': C,
              'kernel': kernel,
              'gamma': gamma,
              'degree': degree,
              'probability': probability
             }

pprint(random_grid)


# In[ ]:


# First create the base model to tune
#svc = svm.SVC(random_state=8)

# Definition of the random search
#random_search = RandomizedSearchCV(estimator=svc,
                                   param_distributions=random_grid,
                                   n_iter=50,
                                   scoring='accuracy',
                                   cv=3, 
                                   verbose=1, 
                                   random_state=8)

# Fit the random search model
#random_search.fit(features_train, labels_train)


# In[55]:


#print("The best hyperparameters from Random Search are:")
#print(random_search.best_params_)
#print("")
##print("The mean accuracy of a model with these hyperparameters is:")
#print(random_search.best_score_)


# In[56]:


C = [.0001, .001, .01, .1]
degree = [3, 4, 5]
gamma = [1, 10, 100]
probability = [True]

param_grid = [
  {'C': C, 'kernel':['linear'], 'probability':probability},
  {'C': C, 'kernel':['poly'], 'degree':degree, 'probability':probability},
  {'C': C, 'kernel':['rbf'], 'gamma':gamma, 'probability':probability}
]

# Create a base model
svc = svm.SVC(random_state=8)

# Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=svc, 
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=cv_sets,
                           verbose=1)

# Fit the grid search to the data
grid_search.fit(features_train, labels_train)


# In[57]:


print("The best hyperparameters from Grid Search are:")
print(grid_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(grid_search.best_score_)


# In[58]:


best_svc = grid_search.best_estimator_
best_svc


# # Model Fit and performance

# In[59]:


best_svc.fit(features_train, labels_train)


# In[60]:


svc_pred = best_svc.predict(features_test)
#svc_pred = best_svc.predict_proba(features_test)


# In[61]:


# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, best_svc.predict(features_train)))


# In[62]:


print("The test accuracy is: ")
print(accuracy_score(labels_test, svc_pred))


# In[63]:


print("Classification report")
print(classification_report(labels_test,svc_pred))


# In[64]:


base_model = svm.SVC(random_state = 8)
base_model.fit(features_train, labels_train)
accuracy_score(labels_test, base_model.predict(features_test))


# In[65]:


best_svc.fit(features_train, labels_train)
accuracy_score(labels_test, best_svc.predict(features_test))


# In[66]:


d = {
     'Model': 'SVM',
     'Training Set Accuracy': accuracy_score(labels_train, best_svc.predict(features_train)),
     'Test Set Accuracy': accuracy_score(labels_test, svc_pred)
}

df_models_svc = pd.DataFrame(d, index=[0])


# In[67]:


Export = df_models_svc.to_json(r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_files\df_models_svc.jsonl',orient='records', lines=True)


# In[68]:


with open(r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_files\best_svc.pickle', 'wb') as output:
    pickle.dump(best_svc, output)


# In[69]:


predictions = best_svc.predict(features_test)


# In[71]:


# Indexes of the test set
index_X_test = X_test.index
# We get them from the original df
df_test = df_process_bbc.loc[index_X_test]
df_test['Category_Predicted'] = predictions
for col in df_test.columns: 
    print(col)
df_test.head(5)


# In[72]:


category_names = {
    0: 'Business',
    1: 'Entertainment',
    2: 'Politics',
    3: 'Sport',
    4: 'Tech'
}
df_test = df_test.replace({'Category_Predicted':category_names})
df_test.head(5)


# In[74]:


df_misclassified = df_test[df_test['Category_code'] != df_test['Category_Predicted']]
print(len(df_test))
print(len(df_misclassified))
df_misclassified.head(3)


# # MIllion test data classification

# In[1]:


import pandas as pd
from elasticsearch import Elasticsearch,helpers
from elasticsearch.helpers import bulk, streaming_bulk
import uuid
es = Elasticsearch([{'host': 'localhost', 'port': '9200'}],timeout=30, max_retries=10, retry_on_timeout=True)
import json
import time
from pandas.io.json import json_normalize


# In[6]:


df = pd.DataFrame()

start = time.time()
page = es.search(
    index='processdata',
    scroll='40m',
    size=10000,
    _source=["Content_Parsed"],
    body={},
    request_timeout=10000
)
sid = page['_scroll_id']
scroll_size = len(page['hits']['hits'])

df = df.append(json_normalize(page['hits']['hits']))
i=1
while (scroll_size > 0):
    #print("scrolling..",i)
    i=i+1
    page = es.scroll(scroll_id = sid, scroll = '40m')
    df = df.append(json_normalize(page['hits']['hits']))
    sid = page['_scroll_id']
    scroll_size = len(page['hits']['hits'])
end = time.time() - start
print(end)
print(len(df))


# In[90]:


import json_lines
#filepath = r'C:\Users\mouni\OneDrive\Desktop\Info VIz\jsonfiles\ppdata.jsonl'
#df = pd.DataFrame()
##for item in json_lines.reader(filepath):
  #  df= pd.read_json(item)
    #obj = json.load(json_data)
#df_news=pd.read_json(filepath,orient='columns')
with open(r'C:\Users\mouni\OneDrive\Desktop\Info VIz\jsonfiles\ppdata.jsonl',encoding="utf8") as json_data:
    df= pd.read_json(json_data,lines=True)
#    print(type(obj))
    #frame = pd.read_json(json_data)


# In[ ]:


X_test_series = pd.read_pickle(r'C:\Users\mouni\OneDrive\Desktop\Info VIz\trained_files\dfpickle.pickle')


# In[7]:


df.head(1)


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer
# Parameter election
ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 300
tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)


# In[14]:


features_test1 = tfidf.transform(df["_source.Content_Parsed"]).toarray()
print(features_test1.shape)


# In[ ]:


svc_pred = best_svc.predict(features_test1)


# In[ ]:


# Indexes of the test set
index_X_test_series = X_test_series.index
print(index_X_test_series)
# We get them from the original df
df_test100 = df_articles_test100.loc[index_X_test_series]

# Add the predictions
df_test100['Category_Predicted'] = svc_pred
df_test100 = df_test100.replace({'Category_Predicted':category_names})
print(df_test100)


# In[ ]:


def get_category_name(category_id):
    for category, id_ in category_codes.items():    
        if id_ == category_id:
            return category
def predict_from_text(text):
    
    # Predict using the input model
    X_series = pd.Series(text)
    features_test2 = tfidf.transform(X_series).toarray()
    prediction_svc = best_svc.predict(features_test2)
    prediction_svc_proba = best_svc.predict_proba(features_test2)
    
    # Return result
    category_svc = get_category_name(prediction_svc)
    d_series = {}
    d_series["Content"] = text
    if(prediction_svc_proba.max()*100 <= 65):
        d_series["Category_Predicted"] = "Others"
    else:
        d_series["Category_Predicted"] = category_svc
    df_series_100.append(d_series)


# In[ ]:


df_series_100 =[]
for i in X_test_series:
    predict_from_text(i)
df_final = pd.DataFrame(df_series_100)
print(df_final)


# In[ ]:


d

