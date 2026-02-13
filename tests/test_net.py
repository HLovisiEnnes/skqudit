'''
------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
------------------------------------------------------------------------------
Tests for the net module.
------------------------------------------------------------------------------
'''

import pytest
import numpy as np
import os

# Local imports
from skqudit.gates import InstructionSet
from skqudit.net import SimpleNet

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
        's1': np.array([[0, 1j], [1j, 0]]),  # Pauli-X
        's2': np.array([[1j, 0], [0, -1j]]),  # Pauli-Z
        's3': np.array([[0, 1], [-1, 0]])  # Pauli-Y
    }
    return InstructionSet(instrs)


@pytest.fixture
def fix_simple_net(fix_instruction_set):
    '''
    Builds a simple net using the Pauli gates.
    '''
    return SimpleNet(fix_instruction_set)


'''
Test net
'''


def test_simple_net_build(fix_simple_net):
    '''
    Tests the build_net method of SimpleNet.
    '''
    net = fix_simple_net
    initial_layers = net.layers
    new_layers = 3
    net.build_net(new_layers)
    assert net.layers == initial_layers + new_layers
    # Check if the number of nodes does not exceed the expcted maximum
    expected_nodes = len(net.instr_set.instrs) ** (initial_layers + new_layers)
    assert len(net.net) <= expected_nodes


def test_save_and_load(fix_instruction_set, fix_simple_net):
    '''
    Tests the save and load methods of SimpleNet.
    '''
    net1 = fix_simple_net
    net1.build_net(3)
    file = 'example_net.csv'
    net1.save(file)

    net2 = SimpleNet(fix_instruction_set)
    net2.load(file)
    os.remove(file)

    # I will only check lengths as objects will have
    # different memory addresses
    assert len(net2.net) == len(net1.net)
    assert net1.net[42].instr == net2.net[42].instr
