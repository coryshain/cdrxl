import os
import shutil
import argparse

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
    D_train = load_data(
        C.X_train_path,
        Y_path=C.Y_train_path,
        series_ids=C.series_ids,
        predictor_columns=C.predictor_columns,
        response_columns=C.response_columns,
        n_backward=C.n_backward,
        n_forward=C.n_forward,
        batch_size=C.batch_size,
        shuffle=True,
    )

    print('Training Set Variance: %.02f' % D_train.mean_var())

    if C.Y_dev_path:
        D_dev = load_data(
            C.X_dev_path,
            Y_path=C.Y_dev_path,
            series_ids=C.series_ids,
            predictor_columns=C.predictor_columns,
            response_columns=C.response_columns,
            n_backward=C.n_backward,
            n_forward=C.n_forward,
            batch_size=C.batch_size,
            shuffle=False,
            X_mean=D_train.X_mean,
            X_sd=D_train.X_sd,
            X_time_mean=D_train.X_time_mean,
            X_time_sd=D_train.X_time_sd,
        )
        print('Dev Set Variance: %.02f' % D_dev.mean_var())
    else:
        D_dev = None

    kwargs = {}
    for kwarg in Config.CDRXL_KWARGS:
        kwargs[kwarg] = getattr(C, kwarg)

    if args.force_restart and os.path.exists(C.outdir):
        ans = None
        while ans not in ('y', 'n'):
            ans = input('You have requested to delete directory %s. Do you want to proceed? y/[n] >>> ' % C.outdir)
            ans = ans.strip()
            if not ans:
                ans = 'n'
        if ans == 'y':
            shutil.rmtree(C.outdir)

    m = CDRXL(D_train, **kwargs)
    m.fit(
        D_train,
        validation_data=D_dev,
    )