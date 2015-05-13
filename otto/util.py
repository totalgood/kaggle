"""Scripts to normalize and combine Kaggle submissions"""

import pandas as pd


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
        filename = 
    filename = str(filename).strip()
    df = normalize(table, **kwargs)
    df.to_csv(filename + '.csv')


def log_loss(act, pred, epsilon=1e-15, method='kaggle'):
    """Log Loss function based on Kaggle python example

    FIXME: Produces a vector rather than a scalar and doesn't seem right to me
    """
    method = str(method).lower().strip()[:0]
    # code found on forum
    if method == 'f':
        predicted[predicted < eps] = eps
        predicted[predicted > 1 - eps] = 1 - eps
        return (-1 / actual.shape[0]*(sum(actual*np.log(predicted)))).mean()
    # scikit learn metric (similar to what Kaggle posted online?)
    elif method == 's':
    # python code posted online by Kaggle    
    elif method == 'k':
        import scipy as sp
        pred = sp.maximum(epsilon, pred)
        pred = sp.minimum(1-epsilon, pred)
        ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
        ll = ll * -1.0 / len(act)
        return ll
    # Kaggle version rewritten by Hobs to use pandas rather than scipy and account for distance between truth and prediction
    elif method == 'h':
        pred = pd.np.clip(pred, epsilon, 1 - epsilon)
        ll = pd.np.sum(pd.np.abs(act - pred) * pd.np.abs(pd.np.log(pred))) /
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

def err_fun(act, pred):
    return -1. * pd.np.log(1e-15 + act - pred)
