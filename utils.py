import numpy as np
import operator
from functools import reduce

def load_params(dir):
    return np.load(dir).item()

def flat_shape(tensor):
    """Return flattened dimension of a sample"""
    s = tensor.get_shape()
    shape = tuple([s[i].value for i in range(0, len(s))])
    return reduce(operator.mul, shape[1:])

def update_collection(name, feature, collection):
    """ Update the appropriate features from different the collection """

    # Flatten samples
    if name is not 'label': # label arrays are already in the correct format
        batch_size = np.shape(feature)[0]
        sample_size = reduce(operator.mul, np.shape(feature)[1:])
        feature = np.reshape(feature, (batch_size, sample_size))

    # Add to the previous array
    if np.shape(collection[name])[0] == 0: # if it's the first feature to be inserted in the collection
        collection[name] = feature
    else:
        collection[name] = np.vstack((collection[name], feature))

    return collection

def collect_features(sess, feed_dict, nodes, labels, collection):
    """ Collect the features from nodes in collection (dictionary) """

    # Generate collection dictionary to store all features if 'collection is empty
    if collection == {}:
        for node in nodes:
            collection.update({node.name: np.array([])})
        collection.update({'label': np.array([])})

    # Extract features
    features = sess.run(nodes, feed_dict=feed_dict)

    # Update collection with the extracted features
    for node, feature in zip(nodes, features):
        collection = update_collection(node.name, feature, collection)
    collection = update_collection('label', labels, collection)

    return collection

def prepare_sequence(names, features, batch_size):
    """
    Transform dictionary of features in sequential input for recurrent networks.

    names: ORDERED list of string with the name of features for sequence
    features: dict of ndarrays -- key = layer name, value = (samples x feature size)
    batch_size: desired batch size for the returned sequence
    :return:
    x: (batch_size, time_steps, input_dim)
    y: (batch_size, num_classes)
    """

    pass

    return x, y
