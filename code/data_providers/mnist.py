import tempfile
import os
import pickle
import random
import gzip
import struct
import array
import tempfile
import functools
import operator

import numpy as np

from data_providers.base_provider import ImagesDataSet, DataProvider
from data_providers.downloader import download_data_url


def parse_idx(fd):
    """Parse an IDX file, and return it as a numpy array.
    Parameters
    ----------
    fd : file
        File descriptor of the IDX file to parse
    endian : str
        Byte order of the IDX file. See [1] for available options
    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file
    1. https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment
    """
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type 0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items, len(data)))

    return np.array(data).reshape(dimension_sizes)


class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass

class MNISTDataSet(ImagesDataSet):
    n_classes = 10

    def __init__(self, images, labels, shuffle, normalization):
        """
        Args:
            images: 4D numpy array
            labels: 2D or 1D numpy array
            shuffle: `bool`, should shuffle data or not
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_channels: substract mean of every chanel and divide each
                    chanel data by it's standard deviation
        """
        self._batch_counter, self.epoch_images, self.epoch_labels = 0, None, None

        self.shuffle = shuffle
        self.images = images
        self.labels = labels
        self.normalization = normalization
        self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle:
            self.epoch_images, self.epoch_labels = self.shuffle_images_and_labels(
                self.images, self.labels)
        else:
            self.epoch_images, self.epoch_labels = self.images, self.labels

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        images_slice = self.epoch_images[start: end]
        labels_slice = self.epoch_labels[start: end]
        # due to memory error it should be done inside batch
        if self.normalization is not None:
            images_slice = self.normalize_images(
                images_slice, self.normalization)
        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice


class MNISTDataProvider(DataProvider):
    def __init__(self, save_path=None, validation_size=None, shuffle=False, normalization=None,
                 one_hot=True, **kwargs):
        """
        Args:
            save_path: `str`
            validation_set: `bool`.
            validation_split: `int` or None
                float: chunk of `train set` will be marked as `validation set`.
                None: if 'validation set' == True, `validation set` will be
                    copy of `test set`
            shuffle: `bool`, should shuffle data or not
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            one_hot: `bool`, return lasels one hot encoded
        """
        self._save_path = save_path
        train_images = []
        train_labels = []
        images, labels = self.get_images_and_labels('train', one_hot)
        train_images.append(images)
        train_labels.append(labels)
        train_images = np.vstack(train_images)
        if one_hot:
            train_labels = np.vstack(train_labels)
        else:
            train_labels = np.hstack(train_labels)
        if validation_size is not None:
            np.random.seed(DataProvider._SEED)
            rand_indexes = np.random.permutation(train_images.shape[0])
            valid_indexes = rand_indexes[:validation_size]
            train_indexes = rand_indexes[validation_size:]
            valid_images, valid_labels = train_images[valid_indexes], train_labels[valid_indexes]
            train_images, train_labels = train_images[train_indexes], train_labels[train_indexes]
            self.validation = MNISTDataSet(
                valid_images, valid_labels, False, normalization)

        self.train = MNISTDataSet(
            train_images, train_labels, shuffle, normalization)

        test_images, test_labels = self.get_images_and_labels('t10k', one_hot)
        self.test = MNISTDataSet(test_images, test_labels, False, normalization)

        if validation_size is None:
            self.validation = self.test


    def get_images_and_labels(self, name_part, one_hot=False):
        image_url = self.data_url + name_part + '-images-idx3-ubyte.gz'
        label_url = self.data_url + name_part + '-labels-idx1-ubyte.gz'
        download_data_url(image_url, self.save_path)
        download_data_url(label_url, self.save_path)
        image_filename = os.path.join(self.save_path, name_part + '-images-idx3-ubyte.gz')
        label_filename = os.path.join(self.save_path, name_part + '-labels-idx1-ubyte.gz')
        # unzip and load to np array
        with gzip.open(image_filename, 'rb') as f:
            images = parse_idx(f)
        print("label file {}".format(label_filename))
        with gzip.open(label_filename, 'rb') as f:
            labels = parse_idx(f)
        n_data = images.shape[0]
        images=images.reshape((n_data, 28, 28, 1))
        #labels.reshape((n_data, 1))
        if one_hot:
            labels = self.labels_to_one_hot(labels)
        print(images.shape)
        print(labels.shape)
        return images, labels

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = os.path.join(tempfile.gettempdir(), 'MNIST')
        return self._save_path

    @property
    def data_url(self):
        return 'http://yann.lecun.com/exdb/mnist/'

    @property
    def data_shape(self):
        return 28, 28, 1
