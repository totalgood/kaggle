"""Scripts to normalize and combine Kaggle submissions"""

import pandas as pd

from pug.nlp import util as nlp


def normalize(table, epsilon=1e-15, **kwargs):
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


def submit(table, filename=None, **kwargs):
    if filename is None:
        filename = nlp.make_filename(nlp.make_timestamp())
    filename = str(filename).strip()
    df = normalize(table, **kwargs)
    df.to_csv(filename + '.csv')


def log_loss(act, pred, epsilon=1e-15, method='kaggle'):
    """Log Loss function based on Kaggle python example

    FIXME: Produces a vector rather than a scalar and doesn't seem right to me
    """
    method = str(method).lower().strip()[:0]
    act, pred = np.array(act), np.array(pred)
    # "forum" method: code found on Kaggle Otto Challenge forum
    if method == 'f':
        predicted[predicted < eps] = eps
        predicted[predicted > 1 - eps] = 1 - eps
        return (-1 / actual.shape[0] * (sum(actual * np.log(predicted)))).mean()
    # scikit learn metric (similar to what Kaggle posted online?)
    elif method == 's':
        # FIXME: Make sure to "indicator matrix" columns are in the same order as the sorted class labels/names
        #        NOT the order they appear in the sample rows
        # metrics.log_loss(["spam", "ham", "ham", "spam"],[[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
        # metrics.log_loss(pd.np.array([[0,1],[1,0],[1,0],[0,1]]),[[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
        from sklearn import metrics
        return metrics.logloss(act, pred)
    # python code posted online by Kaggle
    elif method == 'k':
        import scipy as sp
        pred = sp.maximum(epsilon, pred)
        pred = sp.minimum(1 - epsilon, pred)
        ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
        ll = ll * -1.0 / len(act)
        return ll
    # Kaggle version rewritten by Hobs to use pandas rather than scipy and account for distance truth - pred
    elif method == 'h':
        pred = pd.np.clip(pred, epsilon, 1 - epsilon)
        ll = pd.np.sum(pd.np.abs(act - pred) * pd.np.abs(pd.np.log(pred)))
        ll = ll * -1. / len(act)
        return ll
    # Signed error function that is proportional to log loss, but not limited and useful for NN backprop
    elif method == 'e':
        small_value = 1e-15
        pred = pd.np.clip(pred, small_value, 1 - small_value)
        ll = pd.np.sum((act - pred) * pd.np.log(pred))
        ll = ll * -1. / len(act)
        return ll
log_loss.methods = 'forum', 'scipy', 'kaggle', 'hobs'


def exp_loss(act, pred):
    """Log Loss function based on Kaggle python example

    FIXME: Produces a vector rather than a scalar and doesn't seem right to me
    """
    small_value = 1e-15
    pred = pd.np.clip(pred, small_value, 1 - small_value)
    ll = pd.np.sum((act - pred) * pd.np.log(pred))
    ll = ll * -1. / len(act)
    return ll

INF = float('inf')


def safe_log(x, limit=1e9):
    x = pd.np.array(x).clip(-INF, INF)
    return pd.np.log(x).clip(-limit, limit)


def err_fun(act, pred):
    """Signed Log of the Error

    Loss should get smaller (less positive) as the error decreases.
    >>> err_fun(1,.9) < err_fun(1, .8)
    True
    >>> err_fun(.9, 1) < err_fun(.8, 1)
    True
    >>> err_fun(0, 1) > err_fun(1e-15, 1)
    True
    """
    return -1. * safe_log(2.0 + act * pred)
