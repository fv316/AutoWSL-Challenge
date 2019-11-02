# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:42:29 2019

@author: fvice
"""

"""model"""
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from util import log
from preprocess import sample

X  = np.zeros((77349, 68))
X = np.zeros((40000, 69))
X = np.zeros((5686, 21))
X = np.zeros((368363, 59))
dims = X.shape[0] * X.shape[1]
max(int(1000 * 100000/(dims * np.log(dims))), 1)
max(int(1000 * 100000/(dims * np.log(dims))), 1)

class AutoSSLClassifier:
    def __init__(self, train_time_budget):
        self.iter = 5
        self.label_data = 500
        self.model = None
        self.train_time_budget = train_time_budget
        self.multiplier = 1

    def fit(self, X, y):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4
        }
        
        X_label, y_label, X_unlabeled, y_unlabeled = self._split_by_label(X, y)
        y_n_cnt, y_p_cnt = y_label.value_counts()
        
        dims = X_label.shape[0] * X_label.shape[1]
        self.multiplier = max(int(self.train_time_budget * 1000/(dims**0.65 * np.log(dims))), 1)
        print(self.multiplier)
        
        self.label_data = min(y_n_cnt, y_p_cnt)
        y_n = max(int(self.label_data*(y_n_cnt*1.0/len(y_label))), 1)
        y_p = max(int(self.label_data*(y_p_cnt*1.0/len(y_label))), 1)
        n_bias = max(int((y_p/y_n)**0.5), 1)
        p_bias = max(int((y_n/y_p)**0.5), 1)
        
        for _ in range(int(self.iter * self.multiplier)):
            if X_unlabeled.shape[0] < self.label_data:
                break

            hyperparams = self._hyperopt(X_label, y_label, params)

            X_train, X_val, y_train, y_val = train_test_split(X_label, y_label, test_size=0.15, random_state=1)

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val)

            self.model = lgb.train({**params, **hyperparams}, train_data, int(400 * self.multiplier/2),
                                   valid_data, early_stopping_rounds= int(20 * self.multiplier/2), verbose_eval=0)
                
            y_hat = self.model.predict(X_unlabeled)
            
            if len(set(y_hat)) == 1:
                break
            
            conf = 0.01
            limit = 0.2
            
            idx = np.argsort(y_hat)
            sy_hat = y_hat[idx]
            
            y_n_alt = min(sum(sy_hat < min(sy_hat[0] + conf*n_bias, limit)), y_n)
            y_p_alt = max(min(sum(sy_hat > max(sy_hat[-1] - (1 - conf)*p_bias, 1 - limit)), y_p), 1)
            
            y_p_idx = idx[-y_p:]
            y_n_idx = idx[:y_n]
            y_p_idx = idx[-y_p_alt:]
            y_n_idx = idx[:y_n_alt]
            X_label = pd.concat([X_label, X_unlabeled.iloc[list(y_p_idx) + list(y_n_idx), :]])
            y_label = pd.concat([y_label, pd.Series([1]*len(y_p_idx) + [-1]*len(y_n_idx))])
            X_unlabeled = X_unlabeled.iloc[idx[y_n_alt:-y_p_alt], :]
            X_unlabeled = X_unlabeled.iloc[idx[y_n:-y_p], :]
            
            if min(y_p_alt, y_n_alt) <= 1:
                break

        return self

    def predict(self, X):
        return self.model.predict(X)

    def _split_by_label(self, X, y):
        y_label = pd.concat([y[y == -1], y[y == 1]])
        X_label = X.loc[y_label.index, :]
        y_unlabeled = y[y == 0]
        X_unlabeled = X.loc[y_unlabeled.index, :]
        return X_label, y_label, X_unlabeled, y_unlabeled

    def _hyperopt(self, X, y, params):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=1)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
            "max_depth": hp.choice("max_depth", [2, 3, 4, 5, 6]),
            "num_leaves": hp.choice("num_leaves", np.linspace(50, 200, 11, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.6, 1.0, 0.1),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, int(100 * self.multiplier/2),
                              valid_data, early_stopping_rounds= int(20 * int(self.multiplier/2)), verbose_eval=0)

            score = model.best_score["valid_0"][params["metric"]]

            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=int(2 * self.multiplier/2), verbose=0,
                    rstate=np.random.RandomState(1))

        hyperparams = space_eval(space, best)

        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

        return hyperparams


class AutoPUClassifier:
    def __init__(self, train_time_budget):
        self.iter = 2
        self.models = []
        self.train_time_budget = train_time_budget
        self.multiplier = 1

    def fit(self, X, y):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4
        }

        sample_size = 40000
        dims = sample_size * X.shape[1]
        print(dims)
        print(self.train_time_budget)
        print(X.shape)
        self.multiplier = max(int(self.train_time_budget * 1000/(dims**0.65 * np.log(dims))), 1)
        print(self.multiplier)

        for _ in range(self.iter * self.multiplier):
            x_sample, y_sample = self._negative_sample(X, y)
            X_sample, y_sample = sample(x_sample, y_sample, sample_size)

            hyperparams = self._hyperopt(X_sample, y_sample, params)

            X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=0.15, random_state=1)

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val)
            model = lgb.train({**params, **hyperparams}, train_data, 400,
                              valid_data, early_stopping_rounds=20, verbose_eval=0)
            self.models.append(model)

        return self

    def predict(self, X):
        for idx, model in enumerate(self.models):
            p = model.predict(X)
            if idx == 0:
                prediction = p
            else:
                prediction = np.vstack((prediction, p))
        return np.mean(prediction, axis=0)

    def _negative_sample(self, X, y):
        y_n_cnt, y_p_cnt = y.value_counts()
        y_n_sample = y_p_cnt if y_n_cnt > y_p_cnt else y_n_cnt
        y_sample = pd.concat([y[y == 0].sample(y_n_sample), y[y == 1]])
        x_sample = X.loc[y_sample.index, :]

        return x_sample, y_sample

    def _hyperopt(self, X, y, params):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=1)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
            "max_depth": hp.choice("max_depth", [3, 4, 5, 6]),
            "num_leaves": hp.choice("num_leaves", np.linspace(50, 200, 11, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.6, 1.0, 0.2),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.4, 1.0, 0.2),
            "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 100,
                              valid_data, early_stopping_rounds=20, verbose_eval=0)

            score = model.best_score["valid_0"][params["metric"]]

            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=2, verbose=0,
                    rstate=np.random.RandomState(1))
        hyperparams = space_eval(space, best)

        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

        return hyperparams


class AutoNoisyClassifier:
    def __init__(self, train_time_budget):
        self.iter = 2
        self.models = []
        self.train_time_budget = train_time_budget
        self.multiplier = 1

    def fit(self, X, y):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4
        }

        dims = X.shape[0] * X.shape[1]
        self.multiplier = max(int(self.train_time_budget * 1000/(dims**0.65 * np.log(dims))), 1)
        print(self.multiplier)
        
        X_sample, y_sample = sample(X, y, len(X))
        hyperparams = self._hyperopt(X_sample, y_sample, params)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=1)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train({**params, **hyperparams}, train_data, int(400 * self.multiplier),
                               valid_data, early_stopping_rounds=int(20 * self.multiplier/2), verbose_eval=0)
        self.model.append(model)

        return self

    def predict(self, X):
        for idx, model in enumerate(self.models):
            p = model.predict(X)
            if idx == 0:
                prediction = p
            else:
                prediction = np.vstack((prediction, p))
        return np.mean(prediction, axis=0)

    def _hyperopt(self, X, y, params):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=1)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
            "max_depth": hp.choice("max_depth", [3, 4, 5, 6, 7]),
            "num_leaves": hp.choice("num_leaves", np.linspace(50, 200, 11, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, int(100 * self.multiplier),
                              valid_data, early_stopping_rounds=int(20 * self.multiplier/2), verbose_eval=0)
            score = model.best_score["valid_0"][params["metric"]]

            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals= int(2 * self.multiplier), verbose=0,
                    rstate=np.random.RandomState(1))

        hyperparams = space_eval(space, best)
        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
        return hyperparams
