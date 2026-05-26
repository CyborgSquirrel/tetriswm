import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback
import contextlib as ctl

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")


class _Agg:
    def update(self, item):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class agg:
    class min(_Agg):
        def __init__(self, key=None):
            self._key = key
            self._best_key = None
            self.result = None

        def update(self, item):
            k = self._key(item) if self._key else item
            if self.result is None or k < self._best_key:
                self.result = item
                self._best_key = k

    class max(_Agg):
        def __init__(self, key=None):
            self._key = key
            self._best_key = None
            self.result = None

        def update(self, item):
            k = self._key(item) if self._key else item
            if self.result is None or k > self._best_key:
                self.result = item
                self._best_key = k




def pipe(a, *fns):
    for fn in fns:
        a = fn(a)
    return a


@ctl.contextmanager
def torch_printoptions(**kwargs):
    # torch doesn't expose a getter, so we patch __repr__ options via this trick
    import torch._tensor_str as ts
    old = {
        "threshold": ts.PRINT_OPTS.threshold,
        "precision": ts.PRINT_OPTS.precision,
        "linewidth": ts.PRINT_OPTS.linewidth,
        "sci_mode": ts.PRINT_OPTS.sci_mode,
    }
    torch.set_printoptions(**kwargs)
    try:
        yield
    finally:
        torch.set_printoptions(**old)
