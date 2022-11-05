import os
import pickle
import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa

from cdrxl.util import stderr
from cdrxl.config import Config

class CDRXL:
    STATE_ATTRIBUTES = [x for x in Config.CDRXL_KWARGS] + \
        [
            'n_output',
            'n_pred',
            'epoch'
        ]

    def __init__(
            self,
            cdrxl_dataset,
            **kwargs
    ):

        for kwarg in Config.CDRXL_KWARGS:
            setattr(self, kwarg, kwargs.get(kwarg, Config.CDRXL_KWARGS[kwarg]))

        self.n_output = cdrxl_dataset.n_output()
        self.n_pred = cdrxl_dataset.n_pred()
        self.epoch = 0

        self.built = False
        self.model = self.get_model()

        self.load()

        self.model.summary()

    def __getstate__(self):
        return {x: getattr(self, x) for x in self.STATE_ATTRIBUTES}

    def __setstate__(self, state):
        for x in state:
            setattr(self, x, state[x])

        self.built = False
        if not hasattr(self, 'model'):
            self.model = self.get_model()
            self.load_weights()

    @property
    def model_dir(self):
        return os.path.join(self.outdir, 'model')

    @property
    def model_path(self):
        return os.path.join(self.model_dir, 'm.obj')

    @property
    def weights_path(self):
        return os.path.join(self.model_dir, 'm.hdf5')

    def get_model(self):
        n_pred = self.n_pred
        if not self.built:
            if self.regularizer_scale:
                weight_regularizer = tf.keras.regularizers.L2(self.regularizer_scale)
            else:
                weight_regularizer = None

            inputs = tf.keras.Input(
                shape=[None, n_pred],
                name='X'
            )
            input_mask = tf.keras.Input(
                shape=[None],
                name='X_mask'
            )
            t_delta = tf.keras.Input(
                shape=[None, 1],
                name='t_delta'
            )

            irf = t_delta

            if self.resnet:
                L = tf.keras.layers.Dense(
                    self.n_units_irf,
                    activation=None,
                    kernel_regularizer=weight_regularizer,
                    name='IRF_layer_preresnet'
                )
                irf = L(irf)
                if self.dropout_rate:
                    L = tf.keras.layers.Dropout(
                        self.dropout_rate,
                        name='IRF_dropout_preresnet'
                    )
                    irf = L(irf)

            for _L in range(self.n_layers_irf):
                _irf = irf
                if self.resnet:
                    n_inner = 2
                else:
                    n_inner = 1
                for __L in range(n_inner):
                    if __L == 0:
                        activation = 'gelu'
                    else:
                        activation = None
                    L = tf.keras.layers.Dense(
                        self.n_units_irf,
                        activation=activation,
                        kernel_regularizer=weight_regularizer,
                        name='IRF_layer_%d.%d' % (_L + 1, __L + 1)
                    )
                    irf = L(irf)

                    if self.batch_normalize:
                        L = tf.keras.layers.BatchNormalization(name='IRF_BN_%d.%d' % (_L + 1, __L + 1))
                        irf = L(irf)

                    if self.layer_normalize:
                        L = tf.keras.layers.LayerNormalization(name='IRF_LN_%d.%d' % (_L + 1, __L + 1))
                        irf = L(irf)

                if self.resnet:
                    irf = tf.keras.layers.Add()([_irf, irf])

                if self.dropout_rate:
                    L = tf.keras.layers.Dropout(
                        self.dropout_rate,
                        name='IRF_dropout_%d' % (_L + 1)
                    )
                    irf = L(irf)

            if self.irf_by_predictor:
                irf_n_out = n_pred
            else:
                irf_n_out = 1

            L = tf.keras.layers.Dense(
                irf_n_out,
                activation=None,
                kernel_regularizer=weight_regularizer,
                name='IRF_layer_final'
            )
            irf = L(irf)
            if self.dropout_rate:
                L = tf.keras.layers.Dropout(
                    self.dropout_rate,
                    name='IRF_dropout_final'
                )
                irf = L(irf)

            outputs = inputs * irf

            L = tf.keras.layers.GlobalAveragePooling1D()
            outputs = L(outputs, mask=input_mask)

            if self.resnet:
                L = tf.keras.layers.Dense(
                    self.n_units_ff,
                    activation=None,
                    kernel_regularizer=weight_regularizer,
                    name='FF_layer_preresnet'
                )
                outputs = L(outputs)
                if self.dropout_rate:
                    L = tf.keras.layers.Dropout(
                        self.dropout_rate,
                        name='FF_dropout_preresnet'
                    )
                    outputs = L(outputs)

            for _L in range(self.n_layers_ff):
                _outputs = outputs
                if self.resnet:
                    n_inner = 2
                else:
                    n_inner = 1
                for __L in range(n_inner):
                    if __L == 0:
                        activation = 'gelu'
                    else:
                        activation = None
                    L = tf.keras.layers.Dense(
                        self.n_units_ff,
                        activation=activation,
                        kernel_regularizer=weight_regularizer,
                        name='FF_layer_%d.%d' % (_L + 1, __L + 1)
                    )
                    outputs = L(outputs)

                    if self.batch_normalize:
                        L = tf.keras.layers.BatchNormalization(name='FF_BN_%d.%d' % (_L + 1, __L + 1))
                        outputs = L(outputs)

                    if self.layer_normalize:
                        L = tf.keras.layers.LayerNormalization(name='FF_LN_%d.%d' % (_L + 1, __L + 1))
                        outputs = L(outputs)

                if self.resnet:
                    outputs = tf.keras.layers.Add()([_outputs, outputs])

                if self.dropout_rate:
                    L = tf.keras.layers.Dropout(
                        self.dropout_rate,
                        name='FF_dropout_%d' % (_L + 1)
                    )
                    outputs = L(outputs)

            L = tf.keras.layers.Dense(
                self.n_output,
                activation=None,
                kernel_regularizer=weight_regularizer,
                name='FF_layer_final'
            )
            outputs = L(outputs)

            inputs = [inputs, input_mask, t_delta]

            model = tf.keras.models.Model(
                inputs=inputs,
                outputs=outputs
            )

            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                amsgrad=True
            )

            metrics = [
                tf.keras.metrics.MeanSquaredError(name='mse'),
                tf.keras.metrics.CosineSimilarity(name='sim'),
                # tfa.metrics.RSquare(name='R2')
            ]

            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=metrics
            )

            model.summary()

            return model

    def __call__(self, inputs):
        return self.model(inputs)

    def fit(self, *args, **kwargs):
        if 'epochs' not in kwargs:
            kwargs['epochs'] = self.n_iter
        if 'initial_epoch' not in kwargs:
            kwargs['initial_epoch'] = self.epoch
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = [
                self.get_checkpoint_callback(),
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(self.outdir, 'tensorboard')
                )
            ]
        return self.model.fit(*args, **kwargs)

    def save(self, path=None):
        if path is None:
            m_dir = self.model_dir
            m_path = self.model_path
            w_path = self.weights_path
        else:
            m_dir = os.path.join(path, 'model')
            m_path = os.path.join(m_dir, 'm.obj')
            w_path = os.path.join(m_dir, 'm.hdf5')

        if not os.path.exists(m_dir):
            os.makedirs(m_dir)

        with open(m_path, 'wb') as f:
            pickle.dump(self, f)

        self.model.save(w_path)

    def load(self, path=None):
        if path is None:
            m_path = self.model_path
        else:
            m_dir = os.path.join(path, 'model')
            m_path = os.path.join(m_dir, 'm.obj')

        if os.path.exists(m_path):
            with open(m_path, 'rb') as f:
                m_tmp = pickle.load(f)
            self.model = m_tmp.model
            self.__setstate__(m_tmp.__getstate__())

    def load_weights(self, path=None):
        if path is None:
            w_path = self.weights_path
        else:
            m_dir = os.path.join(path, 'model')
            w_path = os.path.join(m_dir, 'm.hdf5')

        if os.path.exists(w_path):
            stderr('Loading saved weights...\n')
            m_tmp = tf.keras.models.load_model(w_path)
            self.model.set_weights(m_tmp.get_weights())
        else:
            stderr('No checkpoint to load. Keeping initialization...\n')


    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_checkpoint_callback(self, **kwargs):
        return Checkpoint(self, **kwargs)


class Checkpoint(tf.keras.callbacks.Callback):
    def __init__(
            self,
            cdrxl_model,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.cdrxl_model = cdrxl_model

    def on_epoch_end(self, epoch, logs=None):
        self.cdrxl_model.set_epoch(epoch)
        self.cdrxl_model.save()
