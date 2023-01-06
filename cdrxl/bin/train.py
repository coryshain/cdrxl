import os
import shutil
import argparse

from cdrxl.util import stderr
from cdrxl.config import Config
from cdrxl.data import load_data
from cdrxl.model import CDRXL

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Train a CDRXL model.
    ''')
    argparser.add_argument('config_path', help='Path to config (*.yml) file specifying data paths and hyperparameters.')
    argparser.add_argument('-f', '--force_restart', action='store_true', help='Force retraining from initialization. Any existing checkpoints will be overwritten.')
    args = argparser.parse_args()

    C = Config(args.config_path)

    if args.force_restart and os.path.exists(C.outdir):
        ans = None
        while ans not in ('y', 'n'):
            ans = input('You have requested to delete directory %s.\nDo you want to proceed? y/[n] >>> ' % C.outdir)
            ans = ans.strip()
            if not ans:
                ans = 'n'
        if ans == 'y':
            stderr('Deleting previously saved model (%s)\n' % C.outdir)
            shutil.rmtree(C.outdir)
        else:
            stderr('Hard restart canceled. Loading saved model (%s).\n')

    if not os.path.exists(C.outdir):
        os.makedirs(C.outdir)
        shutil.copy2(args.config_path, os.path.join(C.outdir, 'config.yml'))

    stderr('Loading training data...\n')
    D_train = load_data(
        C.X_train_path,
        Y_path=C.Y_train_path,
        series_ids=C.series_ids,
        predictor_columns=C.predictor_columns,
        response_columns=C.response_columns,
        rangf=C.rangf_columns,
        n_backward=C.n_backward,
        n_forward=C.n_forward,
        batch_size=C.batch_size,
        shuffle=True,
    )

    kwargs = {}
    for kwarg in Config.CDRXL_KWARGS:
        kwargs[kwarg] = getattr(C, kwarg)

    stderr('Initializing model...\n')
    m = CDRXL(D_train, **kwargs)

    if C.Y_dev_path:
        stderr('Loading dev data...\n')
        D_dev = load_data(
            C.X_dev_path,
            Y_path=C.Y_dev_path,
            series_ids=C.series_ids,
            predictor_columns=C.predictor_columns,
            response_columns=C.response_columns,
            rangf=m.rangf,
            rangf_map=m.rangf_map,
            n_backward=C.n_backward,
            n_forward=C.n_forward,
            batch_size=C.batch_size,
            shuffle=False,
            X_mean=m.X_mean,
            X_sd=m.X_sd,
            X_time_mean=m.X_time_mean,
            X_time_sd=m.X_time_sd,
            Y_mean=m.Y_mean,
            Y_sd=m.Y_sd,
        )
    else:
        D_dev = None

    stderr('Training Set Size:     %d\n' % D_train.n)
    stderr('Training Set MAE:      %.02f\n' % D_train.mae)
    stderr('Training Set Variance: %.02f\n' % D_train.mean_var)
    if D_dev is not None:
        stderr('Dev Set Size:          %d\n' % D_dev.n)
        stderr('Dev Set MAE:           %.02f\n' % D_dev.mae)
        stderr('Dev Set Variance:      %.02f\n' % D_dev.mean_var)

    stderr('Fitting model...\n')
    m.fit(
        D_train,
        validation_data=D_dev
    )