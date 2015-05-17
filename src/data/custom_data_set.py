# -*- coding: utf-8 -*-

class CustomDataSet(object):
    """
    Representing train, valid or test sets for XOR and SIN functions

    Parameters
    ----------
    data : 2D numpy array

    Attributes
    ----------
    input : list
    label : list
        A labels for the data given in `input`.
    """

    def __init__(self, data):
        # The label of the digits is always the first fields
        self.input = data[:, 1:]
        self.label = data[:, 0]

    def __iter__(self):
        return self.input.__iter__()
