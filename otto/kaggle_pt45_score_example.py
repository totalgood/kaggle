import os
from timeit import default_timer as cpu_time

import pandas as pd
from pandas import np

from sklearn import ensemble, preprocessing
from sklearn.decomposition import PCA
# from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# import sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model, sklearn.pipeline
from sklearn import metrics

# from sklearn_pandas import DataFrameMapper
# from sklearn_pandas import cross_val_score

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', '')

df_train = pd.DataFrame.from_csv(DATA_PATH + "train.csv")

test = pd.read_csv(DATA_PATH + "test.csv")
# only realy need this to make sure you get the Category labels right and in the right order
sample_submission = pd.read_csv(DATA_PATH + "sampleSubmission.csv")
sample_label_set = sample_submission.columns[1:]  # e.g. Class_1, Class_2, ...
sample_ids = sample_submission.id.values
del sample_submission

train_label_set = sample_label_set
train_feature_names = df_train.columns[:-1]  # don't includ the target column label
train_ids = df_train.index.values
train_targets = df_train.target.values
train_features = df_train[train_feature_names].values
del df_train
test = test.drop('id', axis=1)

# transform counts into Term Frequency x Inverse Document Frequency (normalized term frequency) features
tfidf = TfidfTransformer()

print('Computing TFIDF from training data...')
t0 = cpu_time()
train_features_tfidf = tfidf.fit_transform(train_features).toarray()
del train_features
print("Computing the TFIDF took {} sec of the CPU's time.".format(cpu_time() - t0))

print('Transforming the test data using the trained TFIDF...')
test = tfidf.transform(test).toarray()
print('Finished transforming the test data using the trained TFIDF.')

# encode labels as integers 0-8 (from "Class_1", "Class_2", etc)
classification_encoder = preprocessing.LabelEncoder()
print('Transforming labels from text (class names) into an integer ENUM...')
t0 = cpu_time()
train_targets_encoded = classification_encoder.fit_transform(train_targets)
print("Transforming labels took {} sec of the CPU's time.".format(cpu_time() - t0))
assert(all(sample_label_set[i] == 'Class_{}'.format(i+1) == train_targets[np.where(train_targets_encoded == i)[0][0]]
       for i in range(len(sample_label_set))))

# train a random forest classifier
rfc = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=300)
print('Training a random forest on the training set...')
t0 = cpu_time()
#  `train_features_tfidf` = 93 x 60k matrix of term frequencies normalized (divided by) document frequencies
#  `train_targets` = array(['Class_1', 'Class_1', 'Class_1', ..., 'Class_9', 'Class_9', 'Class_9'], dtype=object)

rfc.fit(train_features_tfidf, train_targets)
print("Random Forest took {} sec of the CPU's time.".format(cpu_time() - t0))

# predict on training set
print('Rerunning the predictor to predict the the labels for the {} training set records...'.format(len(train_features_tfidf)))
t0 = cpu_time()
rfc_preds = pd.DataFrame(rfc.predict_proba(train_features_tfidf), index=train_ids, columns=sample_label_set)
print('completed RFC predictions')
train_actual = pd.DataFrame(np.zeros(rfc_preds.shape), index=train_ids, columns=sample_label_set)
for i in range(len(train_actual)):
    train_actual.iloc[i, train_targets_encoded[i]] = 1
ll_rfc = metrics.log_loss(train_actual.values, rfc_preds.values)
print('log loss for Random Forest: {}'.format(ll_rfc))

print("Training set predictions took {} sec of the CPU's time.".format(cpu_time() - t0))

print("Writing a Kaggle submission csv file")
t0 = cpu_time()
# create submission file
submission_df = pd.DataFrame(rfc.predict_proba(test), index=sample_ids, columns=sample_label_set)
submission_df.to_csv('random_forest_270_submission.csv', index_label='id')


###################################################################################################
### PCA
###

print('Decomposing the features in the {} training set records with PCA...'.format(len(train_features_tfidf)))
t0 = cpu_time()
pca = PCA(n_components=10)
pca.fit(train_features_tfidf)
print("PCA took {} sec of the CPU's time.".format(cpu_time() - t0))

train_components = pca.fit_transform(train_features_tfidf)

pca_rfc = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=200)
print('Training a random forest on the Principal Components of the training set...')
t0 = cpu_time()
pca_rfc.fit(train_components, train_targets)
print("Random Forest on PCA features took {} sec of the CPU's time.".format(cpu_time() - t0))

print('Rerunning the predictor to predict the the labels for the {} training set records...'.format(len(train_features_tfidf)))
t0 = cpu_time()
pca_rfc_preds = pd.DataFrame(pca_rfc.predict_proba(train_features_tfidf), index=train_ids, columns=sample_label_set)
ll_pca_rfc = metrics.log_loss(train_actual.values, pca_rfc_preds.values)
print('log loss for PCA random forest: {}'.format(ll_pca_rfc))

print('Plotting a colormap of the first 30 rows in the training predictions matrix')
from pug.nlp.plot import ColorMap
cm = ColorMap(pca_rfc_preds[:30])
cm.show()

submission_df = pd.DataFrame(pca_rfc.predict_proba(test), index=sample_ids, columns=sample_label_set)
submission_df.to_csv('pca_random_forest_300_submission.csv', index_label='id')

# train_preds.to_csv('training_predictions.csv', index_label='id')


# mapper = DataFrameMapper([
#     ([c for c in df_train.columns if c != 'target'], sklearn.preprocessing.StandardScaler())
#     ('target', sklearn.preprocessing.LabelBinarizer()),
#     ])

# trained_and_binned = mapper.fit_transform(df_train)


# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# PCA(copy=True, n_components=2, whiten=False)
# print(pca.explained_variance_ratio_)
