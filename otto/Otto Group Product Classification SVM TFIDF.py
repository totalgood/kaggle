
# coding: utf-8

# In[7]:

import pandas as pd
import numpy as np
from sklearn import svm, feature_extraction, preprocessing


# In[3]:

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample = pd.read_csv('data/sampleSubmission.csv')


# In[4]:

labels = train.target.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
test = test.drop('id', axis=1)


# In[5]:

tfidf = feature_extraction.text.TfidfTransformer()
train = tfidf.fit_transform(train).toarray()
test = tfidf.transform(test).toarray()


# In[6]:

lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)


# In[8]:

svc_clf = svm.SVC(C=10, probability=True).fit(train, labels)


# In[9]:

preds = svc_clf.predict_proba(test)


# In[10]:

preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('benchmark_tfidf.csv', index_label='id')


# In[ ]:



