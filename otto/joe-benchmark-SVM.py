
# coding: utf-8

# In[52]:

import pandas as pd 
from sklearn.svm import SVC
df = pd.DataFrame(pd.read_csv('data/train.csv'))
test = pd.DataFrame(pd.read_csv('data/test.csv'))
sample = pd.DataFrame(pd.read_csv('data/sampleSubmission.csv'))


# In[53]:

category_map = {}
for index, category in enumerate(df['target'].unique()):
    category_map[category] = index
df['target'] = df['target'].map(category_map)
df_data = df.ix[:,1:94] # select feature data
df_target = df.ix[:,94:95] # select target


# In[54]:

df_target = df_target.values
df_target.transpose()
df_target.flatten()


# In[55]:

rbf_svc = SVC(C=10, probability=True).fit(df_data, df_target)


# In[56]:

test = test.ix[:,1:94]


# In[58]:

preds = rbf_svc.predict_proba(test)


# In[60]:

preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('benchmark_test.csv', index_label='id')


# In[ ]:



