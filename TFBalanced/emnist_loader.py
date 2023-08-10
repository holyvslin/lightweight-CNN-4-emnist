#!/usr/bin/env python
#coding: utf-8

import gzip
import numpy

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def _extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D unit8 numpy array [index, y, x].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols)
    data = numpy.multiply(data, 1.0 / 255.0)
    return data


def _dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

#def _extract_labels(f, one_hot=False, num_classes=10):
def _extract_labels(f, one_hot=False, num_classes=47):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D unit8 numpy array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)

    if one_hot:
        labels = _dense_to_one_hot(labels, num_classes)

    return labels



def read_data_sets():

    # MNIST
    #TRAIN_IMAGES = 'data/train-images-idx3-ubyte.gz'
    #TRAIN_LABELS = 'data/train-labels-idx1-ubyte.gz'
    #TEST_IMAGES = 'data/t10k-images-idx3-ubyte.gz'
    #TEST_LABELS = 'data/t10k-labels-idx1-ubyte.gz'

    #EMNIST Digits
    #TRAIN_IMAGES = 'data/emnist/emnist-digits-train-images-idx3-ubyte.gz'
    #TRAIN_LABELS = 'data/emnist/emnist-digits-train-labels-idx1-ubyte.gz'
    #TEST_IMAGES = 'data/emnist/emnist-digits-test-images-idx3-ubyte.gz'
    #TEST_LABELS = 'data/emnist/emnist-digits-test-labels-idx1-ubyte.gz'

    # EMNIST Letters
    #TRAIN_IMAGES = 'data/emnist/emnist-letters-train-images-idx3-ubyte.gz'
    #TRAIN_LABELS = 'data/emnist/emnist-letters-train-labels-idx1-ubyte.gz'
    #TEST_IMAGES = 'data/emnist/emnist-letters-test-images-idx3-ubyte.gz'
    #TEST_LABELS = 'data/emnist/emnist-letters-test-labels-idx1-ubyte.gz'

    # EMNIST Letters+Digist 47 class
    TRAIN_IMAGES = 'data/emnist/emnist-balanced-train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'data/emnist/emnist-balanced-train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 'data/emnist/emnist-balanced-test-images-idx3-ubyte.gz'
    TEST_LABELS = 'data/emnist/emnist-balanced-test-labels-idx1-ubyte.gz'

    local_file = TRAIN_IMAGES
    with open(local_file, 'rb') as f:
        train_images = _extract_images(f)

    local_file = TRAIN_LABELS
    with open(local_file, 'rb') as f:
        train_labels = _extract_labels(f, one_hot=True)

    local_file = TEST_IMAGES
    with open(local_file, 'rb') as f:
        test_images = _extract_images(f)


    local_file = TEST_LABELS
    with open(local_file, 'rb') as f:
        test_labels = _extract_labels(f, one_hot=True)


    print len(train_images)
    print len(test_images)
    print len(train_labels)
    print len(test_labels)

    #print test_labels[0:20]

    for i in range(0,20):
        #print test_labels[0]
        print i, numpy.where(test_labels[i] == 1)






    return  train_images, test_images, train_labels, test_labels



read_data_sets()
