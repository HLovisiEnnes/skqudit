'''
------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
------------------------------------------------------------------------------
Implements the functionality of buckets for LHS method in search.
------------------------------------------------------------------------------
Utils:
    buckenize
    Bucket
------------------------------------------------------------------------------
'''

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product

from .gates import CompiledGate
from .utils import SU_matrix


def buckenize(
        gate: np.ndarray,
        batch: List[np.ndarray],
        bucket_size: float = 1e-2
        ) -> tuple[int]:
    '''
    Creates buckets based on both the real and imaginary parts of the Frobenius
    inner products of the gate with the elements of a batch. Each bucket is
    defined as the integral part of the these values. In practice, this
    splits the space [-d,d]^{2k} into grids and the number of grids is
    defined by the hyperparameter bucket_size.

    Args:
        gate (np.ndarray): Array of size (dim, dim).
        batch (List[np.ndarray]): List of arrays of size (dim, dim).
        bucket_size (float): Bucket size. Defaults to 1e-2.

    Returns:
        tuple[int]: The bucket tuples.
    '''
    buckets = []
    for r in batch:
        z = np.vdot(r, gate)
        buckets.append(int(z.real / bucket_size))
        buckets.append(int(z.imag / bucket_size))

    return tuple(buckets)


class Bucket:
    '''
    Class that implements a LHS method for gates in a search list. It generates
    k random SU matrix and computes their inner products with elements of the
    list of gates.
    By Cauchy-Schwartz
        |<U,R>-<V,R>| = |<U-V,R>| <= ||U-V|| ||R|| = ||U-V|| d**(1/2),
    so close gates will have similar inner products. We use this fact to trim
    the list of possible gates V close to an input gate U, which reduces total
    search time. In practice, this is done by creating buckets (histograms)
    of gates V based on their inner products.

    Parameters:
        search_list (List[CompliedGate] | SimpleNet): The list of gates
            to be searched.
        k (int): The number of random SU matrices to generate. Defualts to 1.
        bucket_size (float): The size of the buckets. The smaller its value the
            more buckets are created. Defaults to 1e-2. If <V, R> is
            uniformily and densely sampled in [-d, d], then the number of
            buckets is 2d/bucket_size.
        seed (int, optional): The seed to generate the matrices. Should only
            be used if k = 1 or for debugging. Defaults to None.

    Public attributes:
        gate_size (int): The dimension of the gate.
        bucket_size (float): The bucket size used.
        r_matrices (List[np.ndarray]): The generated R-matrices.
        bucket_dict (dict): The dictionarry of keys bucket indices and the
            corresponding indices of the gates in the search list.
        search_list (List[CompiledGate]): The list to be searched.
    '''
    def __init__(
            self,
            search_list: List["CompiledGate"],
            k: int = 1,
            bucket_size: float = 1e-2,
            seed: Optional[int] = None

    ) -> None:
        self.gate_size = search_list[0].gate_size
        self.bucket_size = bucket_size
        self.r_matrices = [SU_matrix(self.gate_size, seed=seed)
                           for _ in range(k)]
        self.bucket_dict = defaultdict(list)
        self.search_list = search_list
        self._build_buckets(search_list=self.search_list)

    def _build_buckets(
            self,
            search_list: List["CompiledGate"]
            ) -> None:
        '''
        Makes the buckets dictionary for a given search list. The dictionary
        keys are the quantization of inner products and values are the
        indices of the gates in the search list.

        Args:
            search_list (List[CompiledGates]): The list of compiled gates
                to be searched.

        Returns:
            None
        '''
        for i, gate in enumerate(search_list):
            self.bucket_dict[buckenize(
                gate.total_unitary,
                self.r_matrices,
                self.bucket_size
            )].append(i)

    def search(
            self,
            gate: np.ndarray,
            robustness: int = 0
    ) -> list:
        '''
        Search for elements of the search list that are in the same or in
        a neighboring bucket of a given gate.

        Args:
            gate (np.ndarray): The gate whose neighbors we want to find.
            robustness (int): The side length of the cube around the
                projections of the gate along the R-matrices to consider
                when returning the result buckets. That is, the number of
                returned buckets is (2*robustness + 1)**k. Defaults to 0.
        Returns:
            list: The list of compiled gates in the same or neighboring
                buckets of the given gate. If no other gates are in these
                buckets, returns an empty list.
        '''
        projections = buckenize(
            gate,
            self.r_matrices,
            self.bucket_size
        )

        # The code below is not the cleanest (i.e., the if clause
        # could be avoided), but it avoids useless iterations if robustness
        # is set to 0
        if robustness:
            # We generate a cube on the lattice of integers of side
            # `robustness` around the projections and retrieve the bucket of
            # generated by each tuple of integers on this cube

            # Generate ranges for each component and then use intertools
            ranges = [
                range(k - robustness, k + robustness + 1) for k in projections
                ]
            neighbor_keys = list(product(*ranges))

            # Add to matches elements of each bucket
            matches = []
            for keys in neighbor_keys:
                matches += self.bucket_dict[keys]

        else:
            matches = self.bucket_dict[projections]

        return [self.search_list[idx] for idx in matches]

    def plot_buckets(
            self,
            dimension: int = 0,
            ) -> None:
        '''
        Plots the histogram of elements per bucket.

        Args:
            dimension (int): The dimension of the bucket dictionary
                key to print. Defaults to 0.

        Returns:
            None
        '''
        coord = []
        numb_els = []
        for key, value in self.bucket_dict.items():
            coord.append(key[dimension])
            numb_els.append(len(value))
        plt.bar(coord, numb_els)
        plt.show()
