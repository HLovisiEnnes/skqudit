'''
------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
------------------------------------------------------------------------------
Tests for the sk module.
------------------------------------------------------------------------------
'''

import pytest
import numpy as np

# Local imports
from skqudit.utils import commutator

from skqudit.utils import test_unitarity as check_unitarity
from skqudit.utils import (test_special_unitarity
                           as check_special_unitarity)

from skqudit.gates import InstructionSet
from skqudit.net import SimpleNet
from skqudit.sk import (
    fourier_diagonalization,
    hermitian_commutator_approximation,
    commutator_approximation
)


'''
Fix an instruction set and net
'''


@pytest.fixture
def fix_instruction_set():
    '''
    Builds Pauli matrices to be used as instruction set.
    '''
    # Recall we are using the su(2) basis
    instrs = {
        's1': np.array([[0, 1j], [1j, 0]]),
        's2': np.array([[1j, 0], [0, -1j]]),
        's3': np.array([[0, 1], [-1, 0]])
    }
    return InstructionSet(instrs)


@pytest.fixture
def fix_simple_net(fix_instruction_set):
    '''
    Builds a simple net using the Pauli gates.
    '''
    return SimpleNet(fix_instruction_set)


'''
Test sk functions
'''


def test_fourier_diagonalization():
    '''
    Tests the Fourier diagonalization method.
    '''
    matrix = np.array([[0, 1], [1, 0]])
    transform, U = fourier_diagonalization(matrix)

    # Checks if the transform is unitary and its inverse does change
    # back to matrix
    assert check_unitarity(U)
    assert np.allclose(np.linalg.inv(U) @ transform @ U, matrix, atol=1e-6)


def test_hermitian_commutator_approximation():
    '''
    Tests the linear algebra solution to the commutator relation.
    '''
    H = np.array([[0, 1], [1, 0]])
    F, G = hermitian_commutator_approximation(H)

    # Check if F and G are indeed Hermitian and the commutator
    # relation is preserved
    assert np.allclose(F, F.conj().T, atol=1e-6)
    assert np.allclose(G, G.conj().T, atol=1e-6)
    assert np.allclose(commutator(F, G), 1j*H, atol=1e-6)


def test_commutator_approximation():
    U = np.array([[0, 1], [1, 0]])
    V, W = commutator_approximation(U)

    # Assert special unitarity of V and W
    assert check_special_unitarity(V)
    assert check_special_unitarity(W)
