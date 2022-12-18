import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class Config:
    CDRXL_KWARGS = {
        'outdir': 'cdrxl_model',
        'n_backward': 32,
        'n_forward': 0,
        'recurrent': False,
        'n_units_irf': 128,
        'n_layers_irf': 1,
        'batch_normalize': False,
        'layer_normalize': False,
        'resnet': False,
        'dropout_rate': 0,
        'regularizer_scale': 0,
        'filter_regularizer_scale': 0,
        'learning_rate': 0.001,
        'batch_size': 256,
        'n_iter': 10000,
        'rescale_inputs': True,
    }

    def __init__(self, path):
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=Loader)

        self.X_train_path = config['X_train_path']
        self.X_dev_path = config.get('X_dev_path', None)
        self.X_test_path = config.get('X_test_path', None)

        self.Y_train_path = config['Y_train_path']
        self.Y_dev_path = config.get('Y_dev_path', None)
        self.Y_test_path = config.get('Y_test_path', None)

        self.series_ids = config.get('series_ids', None)
        self.predictor_columns = config.get('predictor_columns', None)
        self.response_columns = config.get('response_columns', None)
        self.rangf_columns = config.get('rangf_columns', None)

        self.outdir = config.get('outdir', Config.CDRXL_KWARGS['outdir'])
        self.n_backward = config.get('n_backward', Config.CDRXL_KWARGS['n_backward'])
        self.n_forward = config.get('n_forward', Config.CDRXL_KWARGS['n_forward'])
        self.recurrent = config.get('recurrent', Config.CDRXL_KWARGS['recurrent'])
        self.n_units_irf = config.get('n_units_irf', Config.CDRXL_KWARGS['n_units_irf'])
        self.n_layers_irf = config.get('n_layers_irf', Config.CDRXL_KWARGS['n_layers_irf'])
        self.batch_normalize = config.get('batch_normalize', Config.CDRXL_KWARGS['batch_normalize'])
        self.layer_normalize = config.get('layer_normalize', Config.CDRXL_KWARGS['layer_normalize'])
        self.resnet = config.get('resnet', Config.CDRXL_KWARGS['resnet'])
        self.dropout_rate = config.get('dropout_rate', Config.CDRXL_KWARGS['dropout_rate'])
        self.regularizer_scale = config.get('regularizer_scale', Config.CDRXL_KWARGS['regularizer_scale'])
        self.filter_regularizer_scale = config.get('filter_regularizer_scale', Config.CDRXL_KWARGS['filter_regularizer_scale'])
        self.learning_rate = config.get('learning_rate', Config.CDRXL_KWARGS['learning_rate'])
        self.batch_size = config.get('batch_size', Config.CDRXL_KWARGS['batch_size'])
        self.n_iter = config.get('n_iter', Config.CDRXL_KWARGS['n_iter'])
        self.rescale_inputs = config.get('rescale_inputs', Config.CDRXL_KWARGS['rescale_inputs'])
