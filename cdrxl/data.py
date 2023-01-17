import re
import math
import numpy as np
import pandas as pd
import tensorflow as tf


class CDRXLDataSet(tf.keras.utils.Sequence):
    def __init__(
            self,
            X,
            Y=None,
            Y_time=None,
            series_ids=None,
            rangf=None,
            rangf_map=None,
            batch_size=128,
            shuffle=True,
            n_backward=32,
            n_forward=32,
            predictor_columns=None,
            response_columns=None,
            center_X=True,
            rescale_X=True,
            center_time=True,
            rescale_time=True,
            X_mean=None,
            X_sd=None,
            X_time_mean=None,
            X_time_sd=None,
            Y_mean=None,
            Y_sd=None
    ):
        self.series_ids = list(series_ids) if series_ids else []
        self.rangf = list(rangf) if rangf else []

        self.has_series = bool(self.series_ids)
        self.has_rangf = bool(self.rangf)
        self.has_Y = Y is not None

        assert self.has_Y or Y_time is not None, 'Either Y or Y_time must be provided.'

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_backward = n_backward
        self.n_forward = n_forward
        self.center_X = center_X
        self.center_X_time = center_time
        self.rescale_X = rescale_X
        self.rescale_X_time = rescale_time

        if predictor_columns:
            self.predictor_columns = predictor_columns
        else:
            self.predictor_columns = []
            for col in X.columns:
                if col not in ['time'] + self.series_ids + self.rangf:
                    self.predictor_columns.append(col)

        if response_columns:
            self.response_columns = response_columns
        elif self.has_Y:
            self.response_columns = []
            for col in Y.columns:
                if col not in ['time'] + self.series_ids + self.rangf:
                    self.response_columns.append(col)
        else:
            self.response_columns = []

        X = X.sort_values(series_ids + ['time'])
        Y = Y.sort_values(series_ids + ['time'])

        self.X = X[self.predictor_columns].fillna(0.)
        self.X_time = X[['time'] + self.series_ids]
        if self.has_Y:
            self.Y = Y[self.response_columns]
            self.ran = Y[self.rangf]
            self.Y_time = Y[['time'] + self.series_ids]
        else:
            self.ran = Y_time[self.rangf]
            self.Y_time = Y_time[['time'] + self.series_ids]

        self.X_src = self.X
        self.X_time_src = self.X_time
        self.Y_time_src = self.Y_time

        if rangf_map is None:
            self.rangf_map = {}
            for rangf in self.rangf:
                self.rangf_map[rangf] = {x: i for i, x in enumerate(sorted(list(self.ran[rangf].unique())))}
        else:
            self.rangf_map = rangf_map
        if X_mean is None:
            self.X_mean = self.X.values.mean(axis=0)
        else:
            self.X_mean = X_mean
        if X_sd is None:
            self.X_sd = self.X.values.std(axis=0)
        else:
            self.X_sd = X_sd
        if Y_mean is None:
            self.Y_mean = self.Y.values.mean(axis=0)
        else:
            self.Y_mean = Y_mean
        if Y_sd is None:
            self.Y_sd = self.Y.values.std(axis=0)
        else:
            self.Y_sd = Y_sd
        if X_time_mean is None:
            self.X_time_mean = self.X_time.time.values.mean()
        else:
            self.X_time_mean = X_time_mean
        if X_time_sd is None:
            self.X_time_sd = self.X_time.time.values.std()
        else:
            self.X_time_sd = X_time_sd

        if self.center_X:
            self.X = self.X - self.X_mean[None, ...]
        if self.rescale_X:
            self.X = self.X / (self.X_sd[None, ...] + 1e-8)
        if self.center_X_time or self.rescale_X_time:
            self.X_time = self.X_time.copy()
            self.Y_time = self.Y_time.copy()
            if self.center_X_time:
                self.X_time['time'] = self.X_time.time - self.X_time_mean
                self.Y_time['time'] = self.Y_time.time - self.X_time_mean
            if self.rescale_X_time:
                self.X_time['time'] = self.X_time.time / (self.X_time_sd + 1e-8)
                self.Y_time['time'] = self.Y_time.time / (self.X_time_sd + 1e-8)

        self.set_idx()
        self.compute_windows()

    def __len__(self):
        return math.ceil(len(self.Y_time) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.ix[idx * self.batch_size:(idx + 1) * self.batch_size]
        Y_time = self.Y_time.iloc[indices]
        bounds = Y_time[['start', 'end']].values
        Y_time = Y_time.time.values

        T = self.n_backward + self.n_forward
        K = len(self.predictor_columns)

        X = np.zeros((len(Y_time), T, K))
        X_time = np.zeros((len(Y_time), T))
        X_mask = np.zeros((len(Y_time), T))
        for i, (s, e) in enumerate(bounds):
            ix = T - (e - s)
            X[i, ix:] = self.X.iloc[s:e]
            X_time[i, ix:] = self.X_time.time.iloc[s:e]
            X_mask[i, ix:] = 1

        if self.has_Y:
            Y = self.Y.iloc[indices]
        else:
            Y = None
        ran_src = self.ran.iloc[indices]
        ran = []
        rangf_n_levels = self.rangf_n_levels
        for rangf in self.rangf:
            _ran = np.zeros((len(ran_src), rangf_n_levels[rangf]))
            _ranix = ran_src[rangf].map(lambda x, gf=rangf: self.rangf_map[gf].get(x, -1))
            offset = np.where(_ranix >= 0, np.ones(len(_ranix)) ,np.zeros(len(_ranix)))
            _ran[np.arange(len(_ranix)), _ranix] = offset
            _ran = np.tile(tf.expand_dims(_ran, axis=-2), [1, self.n_backward + self.n_forward, 1])
            ran.append(_ran)

        return (X, X_mask, X_time, Y_time, ran), Y

    @property
    def n(self):
        return len(self.Y)

    @property
    def n_pred(self):
        return len(self.predictor_columns)

    @property
    def n_output(self):
        return len(self.response_columns)

    @property
    def mean_var(self):
        return np.square(np.std(self.Y.values, axis=0)).mean()

    @property
    def mae(self):
        return np.mean(np.abs(self.Y.values - self.Y.values.mean(axis=0, keepdims=True))).mean()

    @property
    def rangf_n_levels(self):
        return {x: self.ran[x].nunique() for x in self.rangf}

    def compute_windows(self):
        W = self.Y_time.groupby(self.series_ids).apply(self._compute_windows).squeeze()
        W = pd.DataFrame(W.to_list(), columns=['start', 'end'])
        self.Y_time['start'] = W.start.values
        self.Y_time['end'] = W.end.values

    def _compute_windows(self, Y_time):
        Y_time_src = Y_time
        series_id = tuple(Y_time[self.series_ids].values[0])
        Y_time = Y_time.time.values
        series_id = {x: series_id[i] for i, x in enumerate(self.series_ids)}
        sel = np.ones(len(self.X_time), dtype=bool)
        for x in self.series_ids:
            sel &= self.X_time[x] == series_id[x]
        X_time = self.X_time[sel]
        X_time_src = X_time
        X_ix = np.array(X_time.index)
        X_time = X_time.time.values

        B = self.n_backward
        F = self.n_forward
        i = 0
        j = 0

        bound = []

        while len(bound) < len(Y_time):
            _X_time = X_time[i]
            _Y_time = Y_time[j]
            while _X_time <= _Y_time:
                if i < len(X_time):
                    i += 1
                    _X_time = X_time[i]
            s_ix = max(0, i - B + 1)         # Don't go below 0
            s_ix = min(len(X_ix) - 1, s_ix)  # Don't go past series length
            e_ix = max(0, i + F)             # Don't go below 0
            e_ix = min(len(X_ix) - 1, e_ix)  # Don't go past series length
            s = X_ix[s_ix]
            e = X_ix[e_ix] + 1
            assert e - s >= 0, 'End was before start: %s' % Y_time_src.iloc[i]
            assert e - s <= B + F, 'Chunk length too long: %s' % Y_time_src.iloc[i]
            bound.append((s, e))

            j += 1

        return pd.Series(bound)

    def on_epoch_end(self):
        self.set_idx()

    def set_idx(self):
        ix = np.arange(len(self.Y_time))
        if self.shuffle:
            ix = np.random.permutation(ix)
        self.ix = ix



def load_data(
        X_path,
        Y_path=None,
        predictor_columns=None,
        response_columns=None,
        sep=None,
        **kwargs
):
    X = pd.read_csv(X_path, sep=sep)
    Y = pd.read_csv(Y_path, sep=sep)

    if predictor_columns is not None:
        _predictor_columns = []
        for col in predictor_columns:
            if col in X:
                _predictor_columns.append(col)
            else:
                for _col in X:
                    if re.match(col, _col):
                        _predictor_columns.append(_col)
        predictor_columns = _predictor_columns

    if response_columns is not None:
        _response_columns = []
        for col in response_columns:
            if col in Y:
                _response_columns.append(col)
            else:
                for _col in Y:
                    if re.match(col, _col):
                        _response_columns.append(_col)
        response_columns = _response_columns

    return CDRXLDataSet(
        X,
        Y=Y,
        predictor_columns=predictor_columns,
        response_columns=response_columns,
        **kwargs
    )
