"""Scripts to normalize and combine Kaggle submissions

"""
import os
import itertools
from timeit import default_timer as cpu_time
import pandas as pd
np = pd.np

from pug.nlp import util as nlp

from sklearn import metrics
from sklearn import preprocessing
# from sklearn.decomposition import PCA
# from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import scipy as sp

INF = float('inf')
EPSILON = 1e-15


try:
    DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', '')
    SUBMISSIONS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'submissions', '')
except:
    DATA_PATH = os.path.join('data', '')
    SUBMISSIONS_PATH = os.path.join('submissions', '')


def normalize_dataframe(table, epsilon=1e-15, **kwargs):
    """Force all rows of a table (csv file, ndarray, list of lists, DataFrame) to be probabilities

    All values will be floats ranging between epsilon and 1 - epsilon, inclusive.
    All rows will sum to 1
    """
    filename = None
    if isinstance(table, basestring):
        filename = str(table)
        df = pd.DataFrame.from_csv(filename)
    else:
        df = pd.DataFrame(table, **kwargs)
    df = df.clip(epsilon, 1 - epsilon)
    return df.div(df.sum(axis=1), axis=0)


def one_hot(table):
    table = pd.np.array(table)
    if len(table.shape) > 1:
        table = normalize_dataframe(table)
    else:
        table = normalize_dataframe(pd.np.array([list(table)]))
    return 1 * (table > 0.5)


def one_best(table, axis=1):
    """Set the best (largest) value in each row to 1 and all others to 0

    Arguments:
      table (list of lists or DataFrame or np.array): table of values to be "bested".
      axis (int): dimension number passed to np.argmax. 1 = row-wise, 0 = col-wise

    >>> one_best([[.2, .8], [1, 0]])
    array([[0, 1], [1, 0]])
    """
    table = pd.np.array(table)
    mask = (table == table.max(axis=axis))
    table = 0 * table
    table[mask] = 1
    return table


def submit(table, filename=None, path='submissions', **kwargs):
    """Write a CSV file for a Kaggle submittal after normalizing

    If table is a list of file names load each CSV, normalize and average them
    together to create one new submittal csv file.
    """
    if filename is None:
        filename = nlp.make_filename(nlp.make_timestamp())
    filename = str(filename).strip()
    filename = nlp.update_file_ext(filename, ext='.csv')

    if isinstance(table, (list, tuple)) and all(isinstance(s, basestring) for s in table):
        fn = table[0]
        df = normalize_dataframe(pd.DataFrame.from_csv(fn)) * 1. / len(table)
        print(df.describe())
        print('=' * 100)
        for fn in table[1:]:
            df += normalize_dataframe(pd.DataFrame.from_csv(fn)) * 1. / len(table)
            print(df.describe())
            print('-' * 100)
        submit(df, filename=filename, path=path, **kwargs)
    else:
        print('=' * 100)
        print(table.describe())
        df = normalize_dataframe(table, **kwargs)
        print(df.describe())
        df.index.name = 'id'
        try:
            df.to_csv(os.path.join(path, filename))
        except:
            df.to_csv(filename)
    return df.sum()


def log_loss(act, pred, normalizer=normalize_dataframe, epsilon=EPSILON, method='all'):
    """Log Loss function for Kaggle Otto competition

    Surprisingly, each method implemented below produces a different answer

    >>> log_loss([[1,0],[0,1]],[[.9,.1],[.1,.9]])  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    0.105...
    >>> log_loss([[1,0],[0,1]],[[.9,.1],[.2,.8]])  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    0.164...
    >>> log_loss([[1,0],[0,1]],[[.8,.2],[.2,.8]])  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    0.223...
    >>> log_loss([[1,0],[0,1]],[[.8,.2],[.2,.8]], method='k')  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    0.223...
    >>> log_loss([[1,0],[0,1]],[[.8,.2],[.2,.8]], method='f')  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    0.223...
    >>> log_loss([[1,0],[0,1]],[[.8,.2],[.2,.8]], method='o')  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    -0.277...
    >>> log_loss([[1,0],[0,1]],[[.8,0],[0,.8]], method='o')  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    0.044...
    >>> log_loss([[1,0],[0,1]],[[.8,0],[0,.8]], method='f') == log_loss([[1,0],[0,1]],[[.8,.2],[.2,.8]], method='f')
    True
    >>> log_loss([[1,0],[0,1]],[[.8,0],[0,.8]], method='k')
    0.11...
    """
    method = str(method).lower().strip()[:1]
    act = pd.np.array(act, dtype=pd.np.int)
    pred = pd.np.array(pred, dtype=pd.np.float64)
    if normalizer:
        act = pd.np.array(one_best(act))
        pred = pd.np.array(normalizer(pred), dtype=pd.np.float64)
    # "forum" method: code found on Kaggle Otto Challenge forum, with lots of bugs
    if method == 'f':
        pred[pred < epsilon] = epsilon
        pred[pred > 1 - epsilon] = 1 - epsilon
        return -1. * (sum(act * np.log(pred))).mean()
    # scikit learn metric (similar to what Kaggle posted online?)
    elif method == 's':
        # FIXME: Make sure "indicator matrix" columns are in the same order as the sorted class labels/names
        #        NOT the order they appear in the sample rows
        # metrics.log_loss(["spam", "ham", "ham", "spam"],[[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
        # metrics.log_loss(pd.np.array([[0,1],[1,0],[1,0],[0,1]]),[[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
        return metrics.log_loss(act, pred)
    # python code posted online by Kaggle
    elif method == 'k':
        pred = sp.maximum(epsilon, pred)
        pred = sp.minimum(1 - epsilon, pred)
        ll = -1.0 * np.mean(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
        return ll
    # Kaggle version rewritten by Hobs to use pandas rather than scipy and account for distance truth - pred
    elif method == 'h':
        pred = pd.np.clip(pred, epsilon, 1 - epsilon)
        ll = pd.np.sum(pd.np.abs(act - pred) * pd.np.abs(pd.np.log(pred)))
        ll = ll * -1. / len(act)
        return ll
    # "other" method similar to "kaggle" method only no scipy dependency
    # FIXME: Produces a vector rather than a scalar and doesn't seem right to me
    elif method == 'o':
        pred = pd.np.clip(pred, epsilon, 1 - epsilon)
        ll = pd.np.sum((act - pred) * pd.np.log(pred))
        ll = ll * -1. / len(act)
        return ll
    # Signed error function that is proportional to log loss, but not limited and useful for NN backprop
    elif method == 'e':
        return err_fun(act, pred)
    # all methods
    elif method == 'a':
        return [log_loss(act, pred, normalizer=normalizer, epsilon=epsilon, method=m) for m in log_loss.methods]
log_loss.methods = 'kaggle', 'scipy', 'forum', 'other', 'hobs'


def safe_log(x, limit=500):
    """Take log, ensuring that operation is invertable with exp and nonNaN"""
    x = pd.np.clip(x, EPSILON, INF)
    return pd.np.clip(pd.np.log(x), -limit, limit)


def err_fun(act, pred):
    """Signed Log of the Error, assuming act and pred are 0 < x <= 1

    Loss should get smaller (less positive) as the error decreases.
    >>> abs(err_fun(1,.9)) < abs(err_fun(1, .8))
    True
    >>> abs(err_fun(.5,.5-1e-15)) > abs(err_fun(.5,.5-2e-15))
    True
    >>> err_fun(1,.9) > 0
    True
    >>> err_fun(.9,1) < 0
    True
    >>> abs(err_fun(.9, 1)) < abs(err_fun(.8, 1))
    True
    >>> abs(err_fun(0, 1)) > abs(err_fun(2e-15, 1))
    True
    """
    return pd.np.sign(act - pred) * safe_log(1 + pd.np.clip(pd.np.abs(act - pred), EPSILON, 1 - EPSILON))


def binarize_text_categories(df, class_labels=9, target_column='target',
                             encoded_column=None, in_place=False, verbosity=1):
    if isinstance(class_labels, int) and 1e12 > class_labels > 0:
        ndigit = int((class_labels + 1) / 10.) + 1
        class_labels = [('Class_{:0' + str(ndigit) + 'd}').format(i + 1) for i in range(class_labels)]
    if verbosity > 0:
        print('Transforming {} labels from text (class names) into an integer ENUM...'.format(
              len(class_labels)))
        print('class_labels: {}'.format(class_labels))
    t0 = cpu_time()
    if target_column is None:
        target_column = df.columns[-1]
    if isinstance(target_column, int) and -len(df.columns) <= target_column < len(df.columns):
        target_column = df.columns[target_column]
    if encoded_column is None:
        encoded_column = 'encoded_' + str(target_column.strip())
    classification_encoder = preprocessing.LabelEncoder()
    binary_categorizer = preprocessing.OneHotEncoder(categorical_features='all', dtype=float, handle_unknown='error',
                                                     n_values=len(class_labels), sparse=False)
    df[encoded_column] = classification_encoder.fit_transform(df[target_column])
    binary_classes = binary_categorizer.fit_transform(df[encoded_column].values.reshape(len(df), 1))
    if verbosity > 0:
        print("Transforming labels from text took {} sec of the CPU's time.".format(cpu_time() - t0))
        print("Transformed DataFrame.describe():\n{}".format(df.describe()))
    if in_place:
        for i, label in enumerate(class_labels):
            df[label] = binary_classes[:, i]
        return df
    else:
        return pd.DataFrame(binary_classes, columns=class_labels, index=df.index)


def cologloss(paths=None, verbosity=1):
    """Matrix of loglosses analogous to inverse covariance matrix

    "distance" between sumbission CSVs
    """
    if paths is None:
        paths = 'submissions'
    if isinstance(paths, basestring):
        paths = [f['path'] for f in nlp.find_files(paths, ext='csv', level=1)]
    N = len(paths)
    coloss = np.identity(N)
    if verbosity:
        for i, path in enumerate(paths):
            print("{}: {}".format(i, path))
    for row, col in itertools.product(range(N), range(N)):
        # compute for upper diag only, copy to lower diag
        if row < col:
            if verbosity > 1:
                print('Calculating log_loss for {} -> {}'.format(paths[row], paths[col]))
            df_row = pd.DataFrame.from_csv(paths[row])
            df_col = pd.DataFrame.from_csv(paths[col])
            coloss[row, col] = coloss[col, row] = log_loss(df_row, df_col, method='kaggle')
            if verbosity > 0:
                print("{} <-->- {} = {}".format(row, col, coloss[row, col]))
    return coloss
