#!/bin/env python
import numpy as np
from math import log

def cross_entropy(p, q):
     eps = 1e-15
     return -sum([p[i]*log(q[i]+eps) for i in range(len(p))])

def compute_bce_losses(true_labels:list, predictions:list):
    """
    Implementation inspired from
    https://machinelearningmastery.com/cross-entropy-for-machine-learning/
    """
    losses = []
    for i in range(len(true_labels)):
        
        p = true_labels
        q = predictions
        # create the distribution for each event {0, 1}
        expected = [1.0 - p[i], p[i]]
        predicted = [1.0 - q[i], q[i]]

        # calculate cross entropy for the two events
        ce = cross_entropy(expected, predicted)
        losses.append(ce)
    losses = np.array(losses)
    return losses
