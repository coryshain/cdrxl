import os
import pickle
import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa

from cdrxl.util import stderr
from cdrxl.config import Config

class CDRXL:
    KWARG_ATTRIBUTES = [x for x in Config.CDRXL_KWARGS]
    FIXED_ATTRIBUTES = [
            'n_output',
            'n_pred',
            'rangf',
            'rangf_map',
            'rangf_n_levels',
            'X_mean',
            'X_sd',
            'X_time_mean',
            'X_time_sd'
        ]
    MUTABLE_ATTRIBUTES = [
        'epoch'
    ]
    STATE_ATTRIBUTES = KWARG_ATTRIBUTES + FIXED_ATTRIBUTES + MUTABLE_ATTRIBUTES

    def __init__(
            self,
            cdrxl_dataset,
            **kwargs
    ):

        for kwarg in Config.CDRXL_KWARGS:
            setattr(self, kwarg, kwargs.get(kwarg, Config.CDRXL_KWARGS[kwarg]))

        self.n_output = cdrxl_dataset.n_output
        self.n_pred = cdrxl_dataset.n_pred
        self.rangf = cdrxl_dataset.rangf
        self.rangf_map = cdrxl_dataset.rangf_map
        self.rangf_n_levels = cdrxl_dataset.rangf_n_levels
        self.X_mean = cdrxl_dataset.X_mean
        self.X_sd = cdrxl_dataset.X_sd
        self.X_time_mean = cdrxl_dataset.X_time_mean
        self.X_time_sd = cdrxl_dataset.X_time_sd
        self.epoch = 0

        self.built = False
        self.load()

        self.model.summary()

    def __getstate__(self):
        return {x: getattr(self, x) for x in self.STATE_ATTRIBUTES}

    def __setstate__(self, state):
        for x in state:
            setattr(self, x, state[x])

        self.built = False
        self.model = None

    def __call__(self, inputs):
        assert self.built, "Model not yet initialized, cannot be called on data. First run model.build() or model.load()."
        return self.model(inputs)

    @property
    def model_dir(self):
        return os.path.join(self.outdir, 'model')

    @property
    def model_path(self):
        return os.path.join(self.model_dir, 'm.obj')

    @property
    def weights_path(self):
        return os.path.join(self.model_dir, 'm.hdf5')

    @property
    def weights(self):
        return self.model.weights

    def get_model(self):
        n_pred = self.n_pred
        if self.regularizer_scale:
            weight_regularizer = tf.keras.regularizers.L2(self.regularizer_scale)
        else:
            weight_regularizer = None
        if self.filter_regularizer_scale:
            filter_regularizer = tf.keras.regularizers.L1(self.filter_regularizer_scale)
        else:
            filter_regularizer = None

        inputs = tf.keras.Input(
            shape=[None, n_pred],
            name='X'
        )
        input_mask = tf.keras.Input(
            shape=[None],
            name='X_mask'
        )
        X_time = tf.keras.Input(
            shape=[None],
            name='X_time'
        )
        Y_time = tf.keras.Input(
            shape=[],
            name='Y_time'
        )

        ran = []
        ran_embd = []
        for rangf in self.rangf:
            _ran = tf.keras.Input(
                shape=[None, self.rangf_n_levels[rangf]],
                name='ran_%s' % rangf
            )
            ran.append(_ran)
            L = tf.keras.layers.Dense(
                self.n_units_irf,
                activation=None,
                kernel_regularizer=weight_regularizer,
                name='ran_embedding_layer_%s' % rangf
            )
            ran_embd.append(L(_ran))

        t_delta = (Y_time[..., None] - X_time)[..., None]

        if self.filter_regularizer_scale:
            # Conv1D with equal # filters and groups is a hack to apply trainable hadamard filter
            filter = tf.keras.layers.Conv1D(
                n_pred,
                1,
                padding='same',
                groups=n_pred,
                use_bias=False,
                kernel_regularizer=filter_regularizer
            )
            _inputs = filter(inputs)

        else:
            _inputs = inputs

        irf = tf.keras.layers.Concatenate()([_inputs, t_delta] + ran_embd)  # B x T x F

        if self.resnet:
            L = tf.keras.layers.Dense(
                self.n_units_irf,
                activation=None,
                kernel_regularizer=weight_regularizer,
                name='IRF_layer_preresnet'
            )
            irf = L(irf)

            if self.batch_normalize:
                L = tf.keras.layers.BatchNormalization(name='IRF_BN_preresnet')
                irf = L(irf)

            if self.layer_normalize:
                L = tf.keras.layers.LayerNormalization(name='IRF_LN_preresnet')
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

            if self.resnet:
                irf = tf.keras.layers.Add()([_irf, irf])

            if self.layer_normalize:
                L = tf.keras.layers.LayerNormalization(name='IRF_LN_%d' % (_L + 1))
                irf = L(irf)

            if self.dropout_rate:
                L = tf.keras.layers.Dropout(
                    self.dropout_rate,
                    name='IRF_dropout_%d' % (_L + 1)
                )
                irf = L(irf)

        irf_n_out = self.n_output

        L = tf.keras.layers.Dense(
            irf_n_out,
            activation=None,
            kernel_regularizer=weight_regularizer,
            name='IRF_layer_final'
        )

        outputs = L(irf)
        L = tf.keras.layers.GlobalAveragePooling1D()
        outputs = L(outputs, mask=input_mask)

        inputs = [inputs, input_mask, X_time, Y_time, ran]

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

        return model

    def build(self):
        if not self.built:
            self.model = self.get_model()

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
            stderr('Loading saved model from %s...\n' % m_path)
            with open(m_path, 'rb') as f:
                m_tmp = pickle.load(f)
            state = m_tmp.__getstate__()
            for k in state:
                setattr(self, k, state[k])
            if path is not None:
                self.outdir = path
            self.build()
            self.load_weights(path=path)
        elif not self.built:
            stderr('No saved model to load. Initializing new model.')
            self.build()
        else:
            stderr('No saved model to load. Keeping current weights.')

    def load_weights(self, path=None):
        if path is None:
            w_path = self.weights_path
        else:
            w_path = os.path.join(path, 'model', 'm.hdf5')

        if os.path.exists(w_path):
            stderr('Loading saved weights...\n')
            m_tmp = tf.keras.models.load_model(w_path)
            self.copy_weights(m_tmp)
        else:
            stderr('No checkpoint to load. Keeping initialization...\n')

    def copy_weights(self, keras_model):
        # Use fake gradient to let optimizer initialize weights before swapping them.
        # Ripped from https://github.com/keras-team/keras/issues/15298.
        grad_vars = self.model.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        self.model.set_weights(keras_model.get_weights())
        self.model.optimizer.apply_gradients(zip(zero_grads, grad_vars))
        self.model.optimizer.set_weights(keras_model.optimizer.get_weights())

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
