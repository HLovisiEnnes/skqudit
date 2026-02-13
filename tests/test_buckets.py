'''
------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
------------------------------------------------------------------------------
Tests for the buckets module.
------------------------------------------------------------------------------
'''

import pytest
import numpy as np

from skqudit.gates import InstructionSet
from skqudit.net import SimpleNet
from skqudit.utils import (
    Rx,
    Rz,
    su2_matrix
)
from skqudit.buckets import (
    buckenize,
    Bucket
)

'''
Fixes net and gate to approximate.
'''


@pytest.fixture
def fix_instruction_set():
    '''
    Builds instruction set.
    '''
    s1 = Rx(np.sqrt(2))
    s2 = Rz(np.sqrt(3))

    instr = {
        's1': s1,
        's2': s2
        }
    return InstructionSet(instr)


@pytest.fixture
def fix_net(fix_instruction_set):
    '''
    Builds a net for this instruction set.
    '''
    instr_set = fix_instruction_set
    layers = 10
    net = SimpleNet(instr_set=instr_set)
    net.build_net(layers)

    return net


@pytest.fixture
def fix_gate_to_approximate():
    '''
    Fixes a matrix to approximate.
    '''
    return su2_matrix(1, 2, 2)


@pytest.fixture
def fix_pauli_matrices():
    '''
    Fixes the three Pauli matrices.
    '''
    return [np.array([[0, 1j], [1j, 0]]),
            np.array([[1j, 0], [0, -1j]]),
            np.array([[0, 1], [-1, 0]])]


'''
Test functions and class
'''


def test_buckenize(fix_gate_to_approximate, fix_pauli_matrices):
    '''
    Test the buckenize function.
    '''
    bucket_size = 1
    gate = fix_gate_to_approximate
    batch = fix_pauli_matrices
    buckets = buckenize(gate, batch, bucket_size=bucket_size)

    # Test if we have both real and imaginary parts
    assert len(buckets) == 2*len(batch)

    # Test if the indices are correct
    dot_x = int(np.vdot(batch[0], gate).real / bucket_size)
    dot_y = int(np.vdot(batch[1], gate).real / bucket_size)
    dot_z = int(np.vdot(batch[2], gate).real / bucket_size)

    # Frobenius inner-product is real in SU(2)
    assert np.allclose(dot_x, buckets[0], atol=1e-6)
    assert np.allclose(dot_y.real, buckets[2], atol=1e-6)
    assert np.allclose(dot_z.real, buckets[4], atol=1e-6)


def test_bucket(fix_net, fix_gate_to_approximate):
    '''
    Tests the bucket for an example which we already computed priorly.
    '''
    search_list = fix_net.net
    gate_to_approx = fix_gate_to_approximate

    k = 1
    bucket_size = 0.1
    seed = 42
    bucket = Bucket(search_list, bucket_size=bucket_size, k=k, seed=seed)

    # Check if the correct number of elements are in the buckets with and
    # without robustness
    assert len(bucket.search(gate_to_approx, robustness=0)) == 7540
    assert len(bucket.search(gate_to_approx, robustness=1)) == 23003

    # Test if we have correctly implemented the dependences on
    # bucket_size
    bucket_size = 0.01
    bucket = Bucket(search_list, bucket_size=bucket_size, k=k, seed=seed)

    # Decreasing bucket size should decrease the number of gates in
    # that bucket -> finer division of space
    assert len(bucket.search(gate_to_approx, robustness=0)) <= 7540
    assert len(bucket.search(gate_to_approx, robustness=1)) <= 23003
