# -*- coding: utf-8 -*-
"""Script with convenience functions.

These functions enable and simplify the class methods
implemented in this repository.

Written by: Dmitry Duplyakin (dmitry.duplyakin@nrel.gov)
in collaboration with the National Renewable Energy Laboratories.
"""

from numpy import array

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
