import datetime
from sklearn import preprocessing
import CONSTANT
from util import timeit
import pandas as pd
#from sklearn import impute

@timeit
def clean_table(table):
    clean_df(table)


@timeit
def clean_df(df):
    fillna(df)


@timeit
def fillna(df):
    for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        df[c].fillna(-1, inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c].fillna("0", inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)


@timeit
def feature_engineer(df):
    transform_categorical_hash(df)
    #categorical_encoder(df)
    #category_dtype(df)
    transform_datetime(df)


@timeit
def transform_datetime(df):
    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df.drop(c, axis=1, inplace=True)


@timeit
def transform_categorical_hash(df):
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c] = df[c].apply(lambda x: hash(x))


@timeit
def sample(X, y, nrows):
    if len(X) > nrows:
        X_sample = X.sample(nrows, random_state=1)
        y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample


@timeit
def categorical_encoder(df):
    le = preprocessing.LabelEncoder()
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        try:
            le.fit(df[c])
            encoded_feature = le.transform(df[c])
            df[c] = pd.DataFrame(encoded_feature)
        except Exception as ex:
            raise(Exception("Categorial encoder problem: {}".format(str(ex))))


@timeit
def category_dtype(df):
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c] = df[c].astype('category')
