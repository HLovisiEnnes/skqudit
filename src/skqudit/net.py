'''
------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
------------------------------------------------------------------------------
Implements the classes for the gate net.
------------------------------------------------------------------------------
Utils:
    SimpleNet
------------------------------------------------------------------------------
'''

from typing import Union, Optional
import numpy as np
import csv
import random

# Local imports
from .gates import (
    InstructionSet,
    CompiledGate
)

from .search import (
    search
)

from .utils import (
    frobenius_norm,
    progress_report
    )


class SimpleNet:
    def __init__(self, instr_set: "InstructionSet") -> None:
        '''
        Class for implementing nets of gates (in memory).
        It is constructed as a tree in which we recursively add new gates
        to lower nodes, each representing a compiled gate.

        Parameters:
            instr_set (InstructionSet): Instruction set used to build the net.

        Public attributes:
            instr_set (InstructionSet): Instruction set used to build the net.
            net (list): A list of the net's nodes.
            layers (int): The current number of layers.
        '''
        self.instr_set = instr_set
        self._gate_size = instr_set.gate_size

        # Transform all instruction into gates
        self._instrs = [
            CompiledGate(instr, instr_set) for instr in self.instr_set.instrs
        ]

        # Layer specific attributes, these will change after `build_net`
        # Needs the list construction for deep copying
        self.net = [i for i in self._instrs]
        self._cur_layer = self.net

        # For a net built from ground up, this will correspond to the
        # length of the longest instruction, but it does not
        # necessarily need to be it if we load the net
        self.layers = 1

    def build_net(
            self,
            new_layers: int,
            method: str = 'linear',
            checkpoint: Optional[str] = None,
            checkpoint_strategy: Optional[str] = 'layer',
            tol: Optional[float] = 1e-6,
            verbosity: int = 0
            ) -> list["CompiledGate"]:
        '''
        Builds the net's tree using breadth-first.

        Args:
            new_layers (int): The number of new layers to add.
            method (str): The net building method. Options are:
                - "linear: Builds the net without much comparison with
                    earlier nodes. It is faster, but builds a more redundant
                    net.
                - "quadratic": Compares the distance between any new gate
                    with gates previously added to the net. It is much slower
                    (quadratic instead of linear on the number of
                    instructions), but avoids repetitions.
                Defaults to "linear".
            checkpoint (str, optional): The path to save checkpoints. Needs
                to be a csv file. Defaults to None.
            checkpoint_strategy (str, optional): Strategy to save checkpoints.
                Only used if `checkpoint` is not None. Options are:
                - "layer": Saves only the last layer.
                - "full": Saves the full net.
                Defaults to "layer".
            tol (float, optional): The equality tolerance to be
                considered for the quadratic method. Defaults to 1e-6.
            verbosity (int): Verbosity level. Defaults to 0.

        Returns:
            list[CompiledGate]: The list of updated compiled gates.
        '''
        if (checkpoint is not None) and (not checkpoint.endswith('.csv')):
            raise ValueError(
                """`checkpoint` must either be
                None or a string ending with `.csv`."""
                )

        if (
            checkpoint_strategy is not None
            and checkpoint_strategy != 'layer'
            and checkpoint_strategy != 'full'
        ):
            raise ValueError(
                """`checkpoint_strategy` must either be
                None, "layer", or "full"."""
                )

        target_layer = self.layers + new_layers

        if method == 'linear':
            # A while loop should be faster than a recursion here
            while self.layers < target_layer:
                cur_layer = []
                for node in self._cur_layer:
                    # We avoid adding the inverse of the last gate of the node
                    # since that creates a node on an upper layer of the three
                    cur_layer += (
                        node @ gate
                        for gate in self._instrs
                        if gate.instr.swapcase() != node.instr.split('.')[-1]
                    )
                self.net += cur_layer
                self._cur_layer = cur_layer
                self.layers += 1

                if checkpoint:
                    # Deletes all previous content of the csv if saving only
                    # current layer
                    if checkpoint_strategy == 'layer':
                        with open(checkpoint, "w", newline="") as f:
                            pass

                    with open(checkpoint, "w", newline="") as f:
                        writer = csv.writer(f)
                        for gate in self._cur_layer:
                            # Flattens array and transform complex in strings
                            flat = [
                                str(x) for x in gate.total_unitary.flatten()
                                ]
                            row = (
                                gate.instr,
                                ",".join(flat)
                            )
                            writer.writerow(row)

                if verbosity:
                    progress_report(
                        self.layers,
                        target_layer,
                        'layer'
                    )

        elif method == 'quadratic':
            # This goes *very much* against DRY principles, but
            # avoids repetitive condition checks
            # I will make a function to avoid repetition soon
            # (I SWEAR!)
            while self.layers < target_layer:
                cur_layer = []
                for node in self._cur_layer:
                    for gate in self._instrs:
                        new_gate = node @ gate
                        # We will make a flag to indicate whether there was
                        # a gate in the previous constructted lines that
                        # is too close to the current gate, in which case,
                        # we do not add anything to the net
                        flag = True
                        for old_gate in self.net:
                            dis = frobenius_norm(
                                old_gate.total_unitary,
                                new_gate.total_unitary
                                )
                            if dis < tol:
                                flag = False
                                break
                        if flag:
                            cur_layer.append(new_gate)

                self.net += cur_layer
                self._cur_layer = cur_layer
                self.layers += 1

                if checkpoint:
                    if checkpoint_strategy == 'layer':
                        with open(checkpoint, "w", newline="") as f:
                            pass

                    with open(checkpoint, "w", newline="") as f:
                        writer = csv.writer(f)
                        for gate in self._cur_layer:
                            flat = [
                                str(x) for x in gate.total_unitary.flatten()
                                ]
                            row = (
                                gate.instr,
                                ",".join(flat)
                            )
                            writer.writerow(row)

                if verbosity:
                    progress_report(
                        self.layers,
                        target_layer,
                        'layer'
                    )

        else:
            raise ValueError(
                '`method` should be either "linear" or "quadratic".'
            )

        return self.net

    def __len__(self) -> int:
        '''
        Returns current net size.

        Args:
            None

        Returns:
            int: Net size.
        '''
        return len(self.net)

    def search(
        self,
        gate:  np.ndarray,
        distance: float,
        method: str = 'early_exit',
        miim_method: str = 'early_exit',
        scope: str = 'full',
        shuffle: bool = False,
        verbosity: int = 1
        ) -> Union[
            tuple["CompiledGate", float],
            tuple[None, None]
            ]:
        '''
        Searches the net to find an approximation of the given gate within
        distance.

        Args:
            gate (np.ndarray): The gate to be approximated.
            distance (float): The maximum distance for the approximation.
            method (str): The search method to be used. Options are:
                - 'full': Full linear search of the net.
                - 'early_exit': Linear search that stops at the first
                    approximation found.
                - 'meet_in_the_middle': Uses a divide-and-conquer method
                    in which we explore compiled gates of length up to
                    twice the current net's maximum length. This explores
                    longer sequences at the same memory cost, but at quadratic
                    worst case speed cost.
                Defaults to 'early_exit'.
            miim_method (str): Method to be used for the search subroutine of
                meet in the middle. Options are 'early_exit' and 'full'.
                Defaults to 'early_exit'.
            scope (str): Scope of the search. It can be either 'full' or
                'last_layer'.
            shuffle (bool): Whether to shuffle scope. Defaults to False.
            verbosity (int): Verbosity level. Defaults to 1.

        Returns:
            tuple (CompiledGate, float) | tuple(None, float): The compiled
                    gate approximating the target gate within the given
                    distance and the realized distance, or a None if
                    not found.
        '''
        if scope == 'full':
            search_list = self.net
        elif scope == 'last_layer':
            search_list = self._cur_layer
        else:
            raise ValueError(
                '`scope` needs to be either "full" or "last_layer".'
            )

        if shuffle:
            # Deep copies the search list to avoid pointers
            # Requires more memory, but it is safer
            search_list = [i for i in search_list]
            random.shuffle(search_list)

        return search(
            gate=gate,
            search_list=search_list,
            distance=distance,
            method=method,
            miim_method=miim_method
        )

    def save(self, path: str) -> None:
        '''
        Saves the net's list as a csv file.

        Args:
            path (str): The csv's path. Must include the '.csv' extension.

        Returns:
            None
        '''
        if not path.endswith('.csv'):
            raise ValueError('`path` must end with `.csv`.')

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for gate in self.net:
                flat = [
                    str(x) for x in gate.total_unitary.flatten()
                    ]
                row = (
                    gate.instr,
                    ",".join(flat)
                )
                writer.writerow(row)

    def load(self, path: str, in_place: bool = True) -> None:
        '''
        Loads a net from a csv file.

        Args:
            path (str): The csv's path. Must include the '.csv' extension.
            in_place (bool): Whether to substitute the original first
                layer. Defaults to True.

        Returns:
            None
        '''
        if not path.endswith('.csv'):
            raise ValueError('`path` must end with `.csv`.')

        if in_place:
            self.net = []
            self.layers = 0

        with open(path, newline="") as f:
            reader = csv.reader(f)

            for instr, total_unitary_str in reader:
                # Transforms strings of arrays in complex and unflats
                flat = [complex(x.replace(" ", ""))
                        for x in total_unitary_str.split(",")]
                total_unitary = np.array(flat).reshape(self._gate_size,
                                                       self._gate_size)

                self.net.append(
                    CompiledGate._from_operation(
                        instr,
                        self.instr_set,
                        total_unitary=total_unitary
                        )
                )

        self._cur_layer = self.net
        self.layers += 1
