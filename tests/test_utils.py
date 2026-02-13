'''
------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
------------------------------------------------------------------------------
Tests for utils module.
------------------------------------------------------------------------------
'''

import numpy as np
import pytest

from skqudit.utils import (
    frobenius_norm,
    Rx,
    Rz,
    su2_matrix,
    lift
)

'''
Fix matrix
'''


@pytest.fixture
def fix_pauli_y():
    '''
    Defines the Pauli Y matrix with determinant 1.

    Args:
        None

    Return:
        np.ndarray: Pauli-Y matrix.
    '''
    return np.array([
        [0, -1],
        [1, 0]
    ])


'''
Tests
'''


def test_norms(fix_pauli_y):
    '''
    Tests implementation of Frobenius norms.
    '''
    id = np.eye(2)

    # Tests general norm
    assert (np.linalg.norm(id - fix_pauli_y)
            == frobenius_norm(id, fix_pauli_y))


def test_centralizer():
    '''
    Test whether projection to PSU(d) is working.
    '''
    n = 2
    id = np.eye(n)

    # Norm without centralizer should be positive, otherwise zero
    assert frobenius_norm(id, -id)
    assert frobenius_norm(id, -id, centralizer=True) < 1e-6


def test_gate_factory():
    '''
    Tests Rx, Rz, and su_matrix.
    '''
    # Rx and Rz use the Bloch sphere angle, so double
    # cover of su(2)
    assert np.allclose(su2_matrix(-1/2, 0, 0), Rx(1))
    assert np.allclose(su2_matrix(0, 0, -1/2), Rz(1))
    assert np.allclose(su2_matrix(0, 0, 0), np.eye(2))


def test_lift(fix_pauli_y):
    '''
    Tests lift function.
    '''
    lifted = lift(fix_pauli_y, 1)

    assert lifted[2, 2] == 1

    assert np.allclose(lifted[:2, :2], fix_pauli_y)
