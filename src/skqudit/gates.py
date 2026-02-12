'''
------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
------------------------------------------------------------------------------
Implements the classes for dealing with gate compilation.
------------------------------------------------------------------------------
Utils:
    InstructionSet
    CompiledGate
------------------------------------------------------------------------------
'''

import numpy as np
from typing import Optional


class InstructionSet:
    '''
    General class to work with instruction sets.

    Parameters:
        instruction_dict (dict): Dictionary of instructions.
            Assumed to have gate name for key and its matrix for value.
            Gate inverses are not necessary as they are construct by default.
            *All gates are assumed of same size.*
        inverses_provided (bool): Whether the inverses were given in
            instruction_dict. Defaults to False.
            Obs: Recall that inverses of gates are given by case sawpping.

    Public attributes:
        instr_dict (dict): The instruction dictionary.
        instrs (list[str]): The list of instructions.
        operators (list(np.ndarray)): List of operators.
        gate_size (int): The size of the gates.
    '''

    def __init__(
            self,
            instruction_dict: dict,
            inverses_provided: bool = False
            ) -> None:
        self.instr_dict = instruction_dict
        if not (inverses_provided):
            self.add_inverses()
        self.instrs = list(self.instr_dict.keys())
        self.operators = list(self.instr_dict.values())
        self.gate_size = self.operators[0].shape[0]

    def add_inverses(self) -> None:
        '''
        Adds the inverse of each gate available.

        Args:
            None

        Returns:
            None
        '''
        inverse_dict = {
            k.swapcase(): v.conj().T for k, v in self.instr_dict.items()
        }
        self.instr_dict.update(inverse_dict)


class CompiledGate:
    '''
    Main class for a compiled gate.

    Parameters:
        instr (str): The string of instructions representing the gate.
            Different instruction gates are separated by `.` and inverses are
            represented by upper caps.
        instr_set (InstructionSet): The set of available instructions.

    Public attributes:
        instr (str): The compilation instructions.
        instr_set (InstructionSet): The available instruction set.
        gate_size (int): The dimension of the gate.
        operators (list[np.ndarray]): The list of matrices representing
            the compilation instructions. It is assumed to be of the
            same order as the instructions (i.e., in application order).
    '''

    def __init__(
            self,
            instr: str,
            instr_set: "InstructionSet"
            ) -> None:

        self.instr = instr
        self.instr_set = instr_set
        self.gate_size = self.instr_set.gate_size

        # Treats the identity as a gate
        if self.instr == '':
            self.total_unitary = np.eye(self.gate_size)
            self.operators = [self.total_unitary]

        else:
            self.get_operators()
            _ = self.unpack_operators()

    def get_operators(self) -> None:
        '''
        Gets the explicit operators as matrices for a given instruction.

        Args:
            None

        Returns:
            None
        '''
        self.operators = [
            self.instr_set.instr_dict[op] for op in self.instr.split('.')
        ]

    def unpack_operators(self) -> np.ndarray:
        '''
        Takes the product of all operators in the instructions.

        Inputs:
            None

        Returns:
            np.ndarray: The product of all operators in the instructions.
        '''
        # Needs to separate case of only one operator, as `multi_dot`
        # assumes that more than one array is given
        if len(self.operators) == 1:
            self.total_unitary = self.operators[0]
        else:
            self.total_unitary = np.linalg.multi_dot(self.operators[::-1])
        return self.total_unitary

    def __len__(self) -> int:
        '''
        Returns the number of instructions used to constructed the gate.

        Args:
            None

        Returns:
            int: Number of instructions.
        '''
        return len(self.instr.split('.'))

    @classmethod
    def _from_operation(
        cls,
        instr: str,
        instr_set: "InstructionSet",
        total_unitary: Optional[np.ndarray] = None
    ) -> "CompiledGate":

        '''
        Alternative constructor to be used internally by operator.
        Avoids cost of `unpack_operators`.

        Args:
            instr (str): The string of instructions representing the gate.
                Different instruction gates are separated by `.` and
                inverses are represented by upper caps.
            instr_set (InstructionSet): The set of available instructions.
            total_unitary (np.ndarray, optional): If given an array,
                sets it to be the object's unitary.

        Returns:
            CompiledGate: The compiled gate.
        '''
        # The following avoids initialization overhead from the
        # `unpack_operators` function
        cg = cls.__new__(cls)
        cg.instr = instr
        cg.instr_set = instr_set
        cg.gate_size = cg.instr_set.gate_size
        cg.get_operators()

        if not (total_unitary is None):
            cg.total_unitary = total_unitary
        return cg

    def inverse(self, in_place: bool = False) -> "CompiledGate":
        '''
        Gets the instructions for compiling the inverse of the gate.

        Args:
            in_place (bool): If True, substitutes the current gate
            for its inverse.

        Returns:
            CompiledGate: Compilation of the inverse gate.
        '''
        inv_instructions = ".".join([
            char.swapcase() for char in self.instr.split('.')[::-1]
        ])
        inv_operators = [op.conj().T for op in self.operators[::-1]]

        inv_total_unitary = self.total_unitary.conj().T

        if in_place:
            self.instr = inv_instructions
            self.operators = inv_operators
            self.total_unitary = inv_total_unitary
            return self

        cg = CompiledGate._from_operation(inv_instructions, self.instr_set)
        cg.total_unitary = inv_total_unitary

        return cg

    def __matmul__(self, other_gate: "CompiledGate") -> "CompiledGate":
        '''
        Get a compilation of the product of two gates by
        concatenation of inverses.

        Args:
            other_gate (CompiledGate): The gate to right.

        Returns:
            CompiledGate: The compilation of the multiplied gate.
        '''
        if self.instr_set != other_gate.instr_set:
            raise ValueError('The two gates must share instruction sets.')

        cg = CompiledGate._from_operation(
            self.instr + '.' + other_gate.instr,
            self.instr_set
        )

        # Notice the matrix multiplication order opposite to concatenation
        cg.total_unitary = other_gate.total_unitary @ self.total_unitary

        return cg
