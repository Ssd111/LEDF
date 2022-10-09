from __future__ import absolute_import
import warnings

from .market1501 import Market1501
from .msmt17v1 import MSMT17_V1
from .cuhk03 import CUHK03
from .cuhk02 import CUHK02
__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17_V1,
    # 'msmt17v1': MSMT17,
    'cuhk03': CUHK03,
    'cuhk02': CUHK02
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. 
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
