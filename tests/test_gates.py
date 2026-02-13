'''
------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
------------------------------------------------------------------------------
Tests for the gates module.
------------------------------------------------------------------------------
'''

import pytest
import numpy as np

# Local imports
from skqudit.gates import InstructionSet, CompiledGate
# Need to change the name to avoid conflicts with pytest
from skqudit.utils import test_inverse as inverse_checker

'''
Fix an instruction set to use everywhere
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


'''
IntrsuctionSet tests
'''


def test_inverses_instruction_set(fix_instruction_set):
    '''
    Tests the add_inverses method of InstructionSet.
    '''
    instr_set = fix_instruction_set
    for gate in ['s1', 's2', 's3']:
        assert inverse_checker(
            instr_set.instr_dict[gate],
            instr_set.instr_dict[gate.swapcase()]
        )


'''
Gates tests
'''


def test_compiled_gate_total_unitary(fix_instruction_set):
    '''
    Tests the total unitary computation of CompiledGate.
    '''
    instr_set = fix_instruction_set
    instructions = 's1.s2.s1'  # XZX
    compiled_gate = CompiledGate(instructions, instr_set)

    expected_matrix = (
        instr_set.operators[0]
        @ instr_set.operators[1]
        @ instr_set.operators[0]
        )

    assert np.allclose(compiled_gate.total_unitary, expected_matrix)


def test_compiled_gate_multiplication(fix_instruction_set):
    '''
    Tests the matrix multiplication of CompiledGate.
    '''
    instr_set = fix_instruction_set

    instructions1 = 's1.s2.s1'  # XZX
    compiled_gate1 = CompiledGate(instructions1, instr_set)

    instructions2 = 's2.s1.s2'  # ZXZ
    compiled_gate2 = CompiledGate(instructions2, instr_set)

    # Recall that multiplication order is reversed in matrix products
    expected_matrix = (
        instr_set.operators[1]
        @ instr_set.operators[0]
        @ instr_set.operators[1]
        @ instr_set.operators[0]
        @ instr_set.operators[1]
        @ instr_set.operators[0]
        )

    total_gate = compiled_gate1 @ compiled_gate2
    assert np.allclose(total_gate.total_unitary, expected_matrix)


def test_compiled_gate_inverse(fix_instruction_set):
    '''
    Tests the inverse computation of CompiledGate.
    '''
    instr_set = fix_instruction_set

    instructions = 's1.s2.s1'  # XZX
    compiled_gate = CompiledGate(instructions, instr_set)

    inverse_gate = compiled_gate.inverse()

    expected_matrix = np.linalg.inv(compiled_gate.total_unitary)

    assert np.allclose(inverse_gate.total_unitary, expected_matrix)
