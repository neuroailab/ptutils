from __future__ import division, print_function, absolute_import

import h5py
import torch


class DataProvider(object):
    """Interface for all DataProvider subclasses.

    The `DataProvider` class is responsible for parsing incoming requests from
    a `ptutils.Session` and returning the appropriate data.

    To respond appropriately to requests, the `DataProvider` should manage a
    `Dataset` or collection of `Dataset`s (if a session needs to draw upon
    more than one data sources).

    Critically, a `DataProvider` subclass must implement a

        `get_dataloader`

    method that accepts an arbitrary, user-defined request for data and returns a
    valid `torch.utils.data.dataloader` object. This request can be conditioned on
    any aspect of the session's state (e.g. epoch, model accuracy, etc.).

    A `torch.utils.data.dataloader combines a `Dataset` and a `Sampler`, and provides
    single- or multi-process iterators over the dataset. In constructing a dataloader,
    you are free to specify parameters such as batch size, data sampling strategies and the number of
    subprocesses to use for data loading.
    See http://pytorch.org/docs/data.html for more details.
    """
    def __init__(self):
        pass

    def get_dataloader(self):
        """Return a `torch.utils.data.dataloader` given an arbitrary data request."""
        raise NotImplementedError()


class Dataset(torch.utils.data.Dataset):
    """Interface for all Dataset subclasses.

    This class simply extends Pytorch's Dataset class to be able to load data
    in different formats by introducing the notion of a `DataReader` and
    `preprocessor`.
    """
    def __init__(self):
        self.data_source = None
        self.data_reader = None
        self.preprocessor = None

    def __getitem__(self, index):
        """Must return a data item given an index in [0, 1, ..., self.__len__()].

        Recommended implementation:

        1. Using `self.data_reader`, read data from `self.data_source` efficiently.
        2. Preprocess the data using `self.preprocessor`.
        3. Return a data item (e.g. image and label)
        """
        raise NotImplementedError()

    def __len__(self):
        """Return the total size (length) of you dataset."""
        raise NotImplementedError()


class DataReader(object):
    """Interface for DataReader subclasses (e.g. HDF5, TFRecords, etc.)
    - Reads data of a specified format efficiently.
    - Exists completely independent of anything ptutils related.
        (Can use is just to read single data file, if desired.)
    """
    def __init__(self):
        pass

    def read(self, *data_source):
        """Loads data from `data_source` into a dictionary of array-like objects.

        The structure of the data dictionary should reflect intrinsic structure
        of the data source. The `Dataset` class will be responsible for parsing
        this dict.

        Should be overriden by all subclasses.
        """
        raise NotImplementedError()

    def __call__(self, *data_source, **kwargs):
        data_dict = self.read(*data_source, **kwargs)
        return data_dict


class HDF5DataReader(object):

    def __init__(self, hdf5_file_path):
        self.file_path = hdf5_file_path

    def read(self):
        self.data_dict = h5py.File(self.hdf5_file_path, 'r')
    return self.data_dict


class ImageNet(Dataset):

    def __init__(self, data_source, data_reader, preprocessor=None):
        self.data_source = data_source
        self.data_reader = data_reader
        self.preprocessor = preprocessor
