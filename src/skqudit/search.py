'''
------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
------------------------------------------------------------------------------
Implements search methods on lists of gates.
------------------------------------------------------------------------------
Utils:
    linear_search
    meet_in_the_middle
    search
------------------------------------------------------------------------------
'''
from typing import Union, List, Optional
import numpy as np

# Local imports
from .gates import CompiledGate
from .utils import (
    frobenius_norm,
    progress_bar
)
from .buckets import Bucket


def linear_search(
        gate: np.ndarray,
        search_list: List["CompiledGate"],
        distance: float,
        method: str = 'early_stop',
        bucket: Optional["Bucket"] = None,
        bucket_robustness: int = 0,
        verbosity: int = 1,
) -> Union[tuple["CompiledGate", float], tuple[None, None]]:
    '''
    Implements linear search over a list of compiled gates.

    Args:
        gate (np.ndarray): The gate to be approximated.
        search_list (list[CompiledGates]): List of compiled gates to be
            searched.
        distance (float): The maximum distance for the approximation.
        method (str): The search method to be used. Options are:
            - 'full': Full linear search of the net.
            - 'early_exit': Linear search that stops at the first
                approximation found.
            Defaults to 'early_exit'.
        bucket (Bucket, optional): The bucket object used for LSH method.
        bucket_robustness (int): The robustness for the LSH method. It is
            ignored if bucket is None. Defaults to 0.
        verbosity (int): Verbosity level. Defaults to 1.

    Returns:
        tuple (CompiledGate, float) | tuple(None, float): The compiled
                gate approximating the target gate within the given
                distance and the realized distance, or a None if
                not found.
    '''
    target = None
    min_dis = np.inf

    # If a bucket is provided, we trim the search space
    if not (bucket is None):
        search_list = bucket.search(gate, robustness=bucket_robustness)

    for i, node in enumerate(search_list):
        dis = frobenius_norm(gate, node.total_unitary)
        if dis < min_dis:
            min_dis = dis
            if dis <= distance:
                target = node
                if method == 'early_exit':
                    break

        # Nested ifs avoids taking the module each run if no verbosity
        if verbosity:
            if not (i % 1e4):
                progress_bar(i, len(search_list))

    return (target, min_dis)


def meet_in_the_middle_search(
        gate: np.ndarray,
        search_list: List["CompiledGate"],
        distance: float,
        miim_method: str = 'early_stop',
        bucket: Optional["Bucket"] = None,
        bucket_robustness: int = 0,
        verbosity: int = 1
) -> Union[tuple["CompiledGate", float], tuple[None, None]]:
    '''
    Meta method for searching consisting of a divide and conquer approach.
    Significantly decreases memory footprint, but at the cost of worse
    computational time.

    Args:
        gate (np.ndarray): The gate to be approximated.
        search_list (list[CompiledGates]): List of compiled gates to be
            searched.
        distance (float): The maximum distance for the approximation.
        miim_method (str): The search method to be used. Options are:
            - 'full': Full linear search of the net.
            - 'early_exit': Linear search that stops at the first
                approximation found.
            Defaults to 'early_exit'.
        bucket (Bucket, optional): The bucket object used for LSH method.
        bucket_robustness (int): The robustness for the LSH method. It is
            ignored if bucket is None. Defaults to 0.
        verbosity (int): Verbosity level. Defaults to 1.

    Returns:
        tuple (CompiledGate, float) | tuple(None, float): The compiled
                gate approximating the target gate within the given
                distance and the realized distance, or a None if
                not found.
    '''
    # Here we will assume that U \approx W \cdot V, where U is the
    # gate to be approximate. The idea is to first compute of
    # R = U V^dagger for all, V element of the net. Then, we
    # find an approximation of R with W. Finally, return WV.
    # This should make search O(L^2), where L is the total scope,
    # but significantly reduces memory requirements. In particular,
    # if l is the layer value, we recall that L = O(numb_genr**l).

    # Not the most DRY code below, but it's fine for now
    target = None
    min_dis = np.inf

    # Computes R = U V^\dagger
    residuals = [gate @ node.inverse().total_unitary
                 for node in search_list]

    # Now approximate R with W
    for i, R in enumerate(residuals):
        W, dis = search(
            R,
            search_list,
            distance,
            method=miim_method,
            bucket=bucket,
            bucket_robustness=bucket_robustness,
            verbosity=0
            )

        if dis < min_dis:
            min_dis = dis

        # If we were able to converge for some value, we let
        # result = VW (recall the instruction order) is opposite
        # to matrix multiplication order
        # Recall that V = net[i]
        if not (W is None):
            target = search_list[i] @ W
            break

        if verbosity:
            if not (i % 10):
                progress_bar(i, len(search_list))

    return (target, min_dis)


def search(
    gate:  np.ndarray,
    search_list: List["CompiledGate"],
    distance: float,
    method: str = 'early_exit',
    miim_method: str = 'early_exit',
    bucket: Optional["Bucket"] = None,
    bucket_robustness: int = 0,
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
        bucket (Bucket, optional): The bucket object used for LSH method.
        bucket_robustness (int): The robustness for the LSH method. It is
            ignored if bucket is None. Defaults to 0.
        verbosity (int): Verbosity level. Defaults to 1.

    Returns:
        tuple (CompiledGate, float) | tuple(None, float): The compiled
                gate approximating the target gate within the given
                distance and the realized distance, or a None if
                not found.
    '''
    # Linear search methods
    if method == 'full' or method == 'early_exit':
        target, min_dis = linear_search(
            gate=gate,
            search_list=search_list,
            distance=distance,
            method=method,
            bucket=bucket,
            bucket_robustness=bucket_robustness,
            verbosity=verbosity
        )

        # If we were not able to find a neighbor with
        # the faster LSH approach, we do normal linear search
        if target is None:
            target, min_dis = linear_search(
                gate=gate,
                search_list=search_list,
                distance=distance,
                method=method,
                bucket=None,
                verbosity=verbosity
            )

    elif method == 'meet_in_the_middle':
        target, min_dis = meet_in_the_middle_search(
            gate=gate,
            search_list=search_list,
            distance=distance,
            miim_method=miim_method,
            bucket=bucket,
            bucket_robustness=bucket_robustness,
            verbosity=verbosity
        )

    else:
        raise ValueError(
            """`method` should be either "full", "early_exit" or
            "meet_in_the_middle"."""
        )

    if (target is None) and (verbosity > 1):
        print(
            'Not able to find an approximation within the given distance.',
            'Mimimal distance found of: ', min_dis
            )

    return (target, min_dis)
