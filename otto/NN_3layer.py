import os
# import random
from timeit import default_timer as cpu_time

import pandas as pd
# from pandas import np

from sklearn import preprocessing
# from sklearn.decomposition import PCA
# from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.linear_model import LogisticRegression
# import sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model, sklearn.pipeline
from sklearn import metrics


# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVC

from pybrain.tools.customxml.networkwriter import NetworkWriter

from pug.ann import util

# from sklearn_pandas import DataFrameMapper
# from sklearn_pandas import cross_val_score

try:
    DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', '')
except:
    DATA_PATH = os.path.join('data', '')

df = pd.DataFrame.from_csv(DATA_PATH + "train.csv")
df_submission = pd.DataFrame.from_csv(DATA_PATH + "sample_submission.csv")

class_labels = list(df_submission.columns)
feature_labels = list(df.columns[:-1])  # 93 features, last col is target (true classification)
target_labels = list(df.columns[-1:])

assert(set(class_labels) == set(df[df.columns[-1]]))
assert(sorted(class_labels) == list(class_labels))

# transform counts into Term Frequency x Inverse Document Frequency (normalized term frequency) features
tfidf = TfidfTransformer()

print('Computing TFIDF (normalized feature frequency) from training set features...')
t0 = cpu_time()
df_freq = pd.DataFrame(tfidf.fit_transform(df[feature_labels].values).toarray(), columns=feature_labels)
print("Computing the TFIDF took {} sec of the CPU's time.".format(cpu_time() - t0))

classification_encoder = preprocessing.LabelEncoder()
binary_categorizer = preprocessing.OneHotEncoder(categorical_features='all', dtype=float, handle_unknown='error',
                                                 n_values=len(class_labels), sparse=False)

print('Transforming labels from text (class names) into an integer ENUM...')
t0 = cpu_time()
df_freq['target'] = classification_encoder.fit_transform(df.target)
binary_classes = binary_categorizer.fit_transform(df_freq.target.values.reshape(len(df_freq), 1))
binary_target_labels = list(df_freq.columns[-9:])
for i, label in enumerate(class_labels):
    df_freq[label] = binary_classes[:, i]
print("Transforming labels from text took {} sec of the CPU's time.".format(cpu_time() - t0))

ds = util.dataset_from_dataframe(df_freq, normalize=False, delays=[0], inputs=feature_labels, outputs=class_labels,
                                 verbosity=1)
nn = util.ann_from_ds(ds, N_hidden=[31, 12, 9], hidden_layer_type=['Linear', 'Linear', 'Linear'],
                      output_layer_type='Sigmoid', verbosity=1)
trainer = util.build_trainer(nn, ds=ds, verbosity=1)
trainer.trainUntilConvergence(maxEpochs=500, verbose=True)

NetworkWriter.writeToFile(trainer.module, __file__ + '.xml')

# this only works for the linear NN where the output is a float 0..8
# df['target_i'] = df_freq.target
# df['predicted_i'] = np.clip(np.round(trainer.module.activateOnDataset(ds)), 0, 8)
# df['predicted'] = [class_labels[i] for i in df['predicted_i']]
# df.to_csv('training_set_with_predictions')


# columns = feature_labels + target_labels + ['Predicted--{}'.format(outp) for outp in target_labels]
predicted_prob = pd.DataFrame((pd.np.array(trainer.module.activate(i)) for i in trainer.ds['input']),
                              columns=class_labels)


def llfun(act, pred):
    small_value = 1e-15
    pred = pd.np.clip(pred, small_value, 1 - small_value)
    ll = pd.np.sum(act * pd.np.log(pred))
    ll = ll * -1. / len(act)
    return ll

log_loss = llfun(ds['target'], predicted_prob.values)
print('The log loss for the training set was {:.3}'.log_loss)
# df = pd.DataFrame(table, columns=columns, index=df.index[max(delays):])


# ################################################################################
# ########## Predict labels for Validation/Test Set for Kaggle submission
# #

df_test = pd.DataFrame.from_csv(DATA_PATH + "test.csv")
test_ids = df_test.index.values

# transform counts into Term Frequency x Inverse Document Frequency (normalized term frequency) features
tfidf = TfidfTransformer()

print('Transforming the validation set features into a TFIDF frequency matrix...')
df_test = pd.DataFrame(tfidf.fit_transform(df_test[feature_labels].values).toarray(),
                       index=test_ids, columns=feature_labels)
print('Finished transforming the test data using the trained TFIDF.')

# columns = feature_labels + target_labels + ['Predicted--{}'.format(outp) for outp in target_labels]
df_test = pd.DataFrame((pd.np.array(trainer.module.activate(i)) for i in df_test.values),
                       index=test_ids, columns=class_labels)
df_test.index.name = 'id'
df_test.to_csv(__file__ + '.csv')
# #
# ########## Predict labels for Validation/Test Set for Kaggle submission
# ################################################################################

# test = test.drop('id', axis=1)

# # transform counts into Term Frequency x Inverse Document Frequency (normalized term frequency) features
# tfidf = TfidfTransformer()

# print('Computing TFIDF from training data...')
# t0 = cpu_time()
# train_features_tfidf = tfidf.fit_transform(train_features).toarray()
# del train_features
# print("Computing the TFIDF took {} sec of the CPU's time.".format(cpu_time() - t0))

# print('Transforming the test data using the trained TFIDF...')
# test = tfidf.transform(test).toarray()
# print('Finished transforming the test data using the trained TFIDF.')

# # encode labels as integers 0-8 (from "Class_1", "Class_2", etc)
# classification_encoder = preprocessing.LabelEncoder()
# print('Transforming labels from text (class names) into an integer ENUM...')
# t0 = cpu_time()
# train_targets_encoded = classification_encoder.fit_transform(train_targets)
# print("Transforming labels took {} sec of the CPU's time.".format(cpu_time() - t0))
# assert(all(sample_label_set[i] == 'Class_{}'.format(i+1) == train_targets[np.where(train_targets_encoded == i)[0][0]]
#        for i in range(len(sample_label_set))))

# # train a random forest classifier
# rfc = RandomForestClassifier(n_jobs=-1, n_estimators=300)
# print('Training a random forest on the training set...')
# t0 = cpu_time()
# #  `train_features_tfidf` = 60k x 93 matrix of term frequencies normalized (divided by) document frequencies
# #  `train_targets` = array(['Class_1', 'Class_1', 'Class_1', ..., 'Class_9', 'Class_9', 'Class_9'], dtype=object)

# rfc.fit(train_features_tfidf, train_targets)
# print("Random Forest took {} sec of the CPU's time.".format(cpu_time() - t0))

# # predict on training set
# print('Rerunning the predictor to predict the the labels for the {} training set records...'.format(
#        len(train_features_tfidf)))
# t0 = cpu_time()
# rfc_preds = pd.DataFrame(rfc.predict_proba(train_features_tfidf), index=train_ids, columns=sample_label_set)
# print('completed RFC predictions')
# train_actual = pd.DataFrame(np.zeros(rfc_preds.shape), index=train_ids, columns=sample_label_set)
# for i in range(len(train_actual)):
#     train_actual.iloc[i, train_targets_encoded[i]] = 1
# ll_rfc = metrics.log_loss(train_actual.values, rfc_preds.values)
# print('log loss for Random Forest: {}'.format(ll_rfc))
# print("Predictions on training set features took {} sec of the CPU's time.".format(cpu_time() - t0))

# print("Writing a Kaggle submission csv file")
# t0 = cpu_time()
# # create submission file
# submission_df = pd.DataFrame(rfc.predict_proba(test), index=sample_ids, columns=sample_label_set)
# submission_df.to_csv('random_forest_300_submission.csv', index_label='id')


# # ###################################################################################################
# # ### PCA
# # ###

# # print('Decomposing the features in the {} training set records with PCA...'.format(len(train_features_tfidf)))
# # t0 = cpu_time()
# # pca = PCA(n_components=10)
# # train_features_tfidf_pc = pca.fit_transform(train_features_tfidf)
# # print("PCA took {} sec of the CPU's time.".format(cpu_time() - t0))

# # pca_rfc = RandomForestClassifier(n_jobs=-1, n_estimators=200)
# # print('Training a random forest on the Principal Components of the training set...')
# # t0 = cpu_time()
# # pca_rfc.fit(train_features_tfidf_pc, train_targets)
# # print("Random Forest on PCA features took {} sec of the CPU's time.".format(cpu_time() - t0))

# # print('Rerunning the predictor to predict the labels for the {} training set records...'.format(
#          len(train_features_tfidf)))
# # t0 = cpu_time()
# # #  `train_features_tfidf` = 60k x 93 matrix of term frequencies normalized (divided by) document frequencies
# # #  model has only 10 features (principle components)
# # #  so need to xform the features to principle components
# # pca_rfc_preds = pca_rfc.predict_proba(train_features_tfidf_pc)
# # print('PCA predictions are of shape {}'.format(pca_rfc_preds.shape))
# # print('len(train_ids) = len(pca_rfc_preds) :: {} = {}'.format(len(train_ids), len(pca_rfc_preds)))
# # print('len(sample_label_set) = {}'.format(len(sample_label_set)))
# # pca_rfc_preds = pd.DataFrame(pca_rfc_preds, index=train_ids, columns=sample_label_set)
# # ll_pca_rfc = metrics.log_loss(train_actual.values, pca_rfc_preds.values)
# # print('log loss for PCA random forest: {}'.format(ll_pca_rfc))

# # print('Plotting a colormap of the first 30 rows in the training predictions matrix')
# # from pug.nlp.plot import ColorMap
# # cm = ColorMap(pca_rfc_preds[:30])
# # cm.show()

# # submission_df = pd.DataFrame(pca_rfc.predict_proba(test), index=sample_ids, columns=sample_label_set)
# # submission_df.to_csv('pca_random_forest_300_submission.csv', index_label='id')
