'''
------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
------------------------------------------------------------------------------
Implementation of the Solovay-Kitaev algorithm.
------------------------------------------------------------------------------
Utils:
    fourier_diagonalization
    hermitian_commutator_approximation
    commutator_approximation
    solovay_kitaev_dev
    solovay_kitaev
------------------------------------------------------------------------------
'''

from scipy.linalg import expm, logm
import numpy as np
import random
from typing import List, Union, Optional

# Local imports
from .net import SimpleNet
from .gates import CompiledGate
from .utils import (
    test_special_unitarity,
    commutator,
    frobenius_norm
)
from .search import search
from .buckets import Bucket


def fourier_diagonalization(
        matrix: np.ndarray,
        enforce_hermitian: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Computes the Fourier diagonalization of a Hermitian matrix. This is
    important in the Dawrson-Nielsen method as it forces the Hermitian
    matrix in a basis in which its diagonal is zero (on top of traceless,
    which is naturally true for su(d)).

    Args:
        matrix (np.ndarray): The matrix.
        enforce_hermitian (bool): Project the matrix into
            Hermitian matrices for numerical stability.
            Defaults to True.

    Returns:
        np.ndarray: The diagonalized unitary.
        np.ndarray: The change of basis matrix.
    '''
    dim = matrix.shape[0]

    if enforce_hermitian:
        matrix = (matrix + matrix.conj().T) / 2

    _, V = np.linalg.eigh(matrix)
    W = np.fft.fft(np.eye(dim)) / np.sqrt(dim)
    U = W @ V.conj().T
    return (U @ matrix @ U.conj().T, U)


def hermitian_commutator_approximation(
        H: np.ndarray,
        scale: float = 1.0,
        verbosity: int = 0
        ) -> tuple[np.ndarray, np.ndarray]:
    '''
    Compute the approximation of the hermitian commutator as explained in
    page 12 of Dawson and Nilsen.

    Args:
        H (np.ndarray): The Hermitian matrix to be approximated.
        scale (float): Scaling factor for the Hermitian normalization.
            Scale smaller than 1 makes the norm of F smaller, whereas
            scale larger than 1 makes the norm of G smaller.
            Defaults to 1.0.
        verbosity(int): Verbosity level. Defaults to 0.

    Returns:
        The approximation of the matrices F and G.
    '''
    dim = H.shape[0]

    H, U = fourier_diagonalization(H, enforce_hermitian=True)

    # If dim is even, we do eigenvalues (-d/2, ..., -1,, 1, ..., d/2)
    # else (-(d-1)/2, ..., -1, 0 , 1, ..., (d-1)/2), so G is traceless
    if dim % 2:
        diag_g = list(range(int(-(dim-1)/2), int((dim-1)/2) + 1))
    else:
        diag_g = (
            list(range(int(-(dim - 1)/2) - 1, 0))
            + list(range(1, int((dim)/2 + 1)))
        )

    G = np.diag(diag_g).astype(float)
    F = np.zeros_like(H, dtype=complex)
    for i in range(dim):
        for j in range(dim):
            if i != j:
                F[i, j] = 1j * H[i, j] / (G[j, j] - G[i, i])

    # Rescale norms to agree with equation 25 of Dawson and Nielsen
    normalizer = (
        np.sqrt((dim - 1) / 2) / (np.sqrt(np.linalg.norm(H)) * dim**(1/4))
        ) * scale
    F *= normalizer
    G *= 1 / normalizer

    if verbosity:
        print('Diagonalized H:\n', H)
        print('Computed F (in diagonal basis):\n', F)
        print('Computed G (in diagonal basis):\n', G)
        print('Norms: ||H|| =', np.linalg.norm(H),
              ', ||F|| =', np.linalg.norm(F),
              ', ||G|| =', np.linalg.norm(G))

    F, G = U.conj().T @ F @ U, U.conj().T @ G @ U

    if verbosity:
        print('Computed F (in normal basis):\n', F)
        print('Computed G (in normal basis):\n', G)
        H = U.conj().T @ H @ U
        print('Computed brackets: [F, G] =\n', commutator(F, G))
        print('Original H:\n', 1j*H)

    return F, G


def commutator_approximation(
        unitary: np.ndarray,
        verbosity: int = 0,
        scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Compute the approximation of the commutator as explained in
    page 12 of Dawson and Nilsen.

    Args:
        unitary (np.ndarray): The unitary matrix to be approximated.
        verbosity (int): Verbosity level. Defaults to 0.
        scale (float): Scaling factor for the Hermitian normalization.
            Scale smaller than 1 makes the norm of F smaller, whereas
            scale larger than 1 makes the norm of G smaller.
            Defaults to 1.0.

    Returns:
        tuple(): The approximation of the matrices V and W.
    '''
    # This ensures that H is skew-Hermitian
    H = logm(unitary)
    F, G = hermitian_commutator_approximation(1j * H,
                                              scale=scale,
                                              verbosity=verbosity)
    V = expm(1j * F)
    W = expm(1j * G)

    if verbosity > 1:
        comm = V @ W @ V.conj().T @ W.conj().T
        approx_distance = frobenius_norm(unitary, comm)
        print('Total commutator:', comm)
        print('Approximation distance:', approx_distance)

    return V, W


def solovay_kitaev_dev(
        gate: np.ndarray,
        depth: int,
        search_list: List["CompiledGate"],
        epsilon_0: float = 1,
        scale: float = 1.0,
        method: str = 'early_exit',
        miim_method: str = 'early_exit',
        bucket: Optional["Bucket"] = None,
        bucket_robustness: int = 0,
        verbosity: int = 1,
        _hist: list = []
        ) -> CompiledGate:
    '''
    Raw implementation of Solovay-Kitaev based on Dawson and Nielsen.
    Refer to `solovay_kitaev` for user-friendly function.

    Args:
        gate (np.ndarray): Target gate to approximate.
        depth (int): Depth of recursion.
        search_list (List[CompiledGate]): List of compiled gates.
        epsilon_0 (float): The threshold distance for depth 0 net search.
            Defaults to 0.2.
        scale (float): Scaling factor for the Hermitian normalization.
            Scale smaller than 1 makes the norm of F smaller, whereas
            scale larger than 1 makes the norm of G smaller.
            Defaults to 1.0.
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
        verbosity(int): Verbosity level. Defaults to 1.

    Returns:
        CompiledGate: The compiled gate.
    '''
    if depth == 0:
        # Assert scpecial unitarity of gate
        if not test_special_unitarity(gate):
            raise ValueError('`gate` needs to be special unitary.')

        approx_zero = search(
            gate=gate,
            search_list=search_list,
            distance=epsilon_0,
            method=method,
            miim_method=miim_method,
            bucket=bucket,
            bucket_robustness=bucket_robustness)[0]
        if approx_zero is None:
            raise ValueError(
                'Not able to find approximation with the desired accuracy.'
                )
        return approx_zero

    else:
        prev_approx = solovay_kitaev_dev(
            gate, depth - 1,
            search_list,
            epsilon_0=epsilon_0,
            scale=scale,
            miim_method=miim_method,
            method=method,
            bucket=bucket,
            bucket_robustness=bucket_robustness,
            verbosity=verbosity,
            _hist=_hist
            )

        error = frobenius_norm(prev_approx.total_unitary, gate)
        len_prev = len(prev_approx)
        _hist.append((depth - 1, error, len_prev))

        if verbosity > 1:
            print('Depth', depth - 1)
            print('Approxiation error:', error)
            print('Number of gates:', len_prev)

        Delta = gate @ prev_approx.total_unitary.conj().T
        V, W = commutator_approximation(
            Delta,
            scale=scale
        )

        V_prev = solovay_kitaev_dev(
            V,
            depth - 1,
            search_list,
            epsilon_0=epsilon_0,
            scale=scale,
            method=method,
            miim_method=miim_method,
            bucket=bucket,
            bucket_robustness=bucket_robustness
            )
        W_prev = solovay_kitaev_dev(
            W,
            depth - 1,
            search_list,
            epsilon_0=epsilon_0,
            scale=scale,
            method=method,
            miim_method=miim_method,
            bucket=bucket,
            bucket_robustness=bucket_robustness
            )

        if verbosity > 2:
            V_prev_uni = V_prev.total_unitary
            W_prev_uni = W_prev.total_unitary
            print('\nError V:', frobenius_norm(V_prev_uni, V))
            print('Error W:', frobenius_norm(W_prev_uni, W))

        return (
            prev_approx @ W_prev.inverse() @ V_prev.inverse() @ W_prev @ V_prev
            )


def solovay_kitaev(
        gate: np.ndarray,
        depth: int,
        net: "SimpleNet",
        epsilon_0: float = 1,
        scale: float = 1.0,
        method: str = 'early_exit',
        miim_method: str = 'early_exit',
        scope: str = 'full',
        shuffle: bool = False,
        bucket_params: Optional[dict] = None,
        return_history: bool = False,
        verbosity: int = 1
        ) -> Union[CompiledGate, List[tuple]]:
    '''
    Implementation of Solovay-Kitaev based on Dawson and Nielsen.

    Args:
        gate (np.ndarray): Target gate to approximate.
        depth (int): Depth of recursion.
        net (SimpleNet): Structure to approximate a gate at depth 0.
        epsilon_0 (float): The threshold distance for depth 0 net search.
            Defaults to 0.2.
        scale (float): Scaling factor for the Hermitian normalization.
            Scale smaller than 1 makes the norm of F smaller, whereas
            scale larger than 1 makes the norm of G smaller.
            Defaults to 1.0.
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
        random_projection_params (dict | None): Parameters to use in
            the random projection. If None, random projection is not
            used. Defaults to None.
        scope (str): Scope of the search. It can be either 'full' or
            'last_layer'.
        shuffle (bool): Whether to shuffle before doing the search.
            Defaults to False.
        bucket_params (dict, optional): The parameters to be used for
            the LSH bucket method. Assumed to be a dicitonary of keys
            'k', 'bucket_size', and 'bucket_robustness'. If None, bucket
            method is not used. Defaults to None.
        return_history (bool): Whether to return the algorithm's history.
            Defaults to False.
        verbosity (int): Verbosity level. Defaults to 1.

    Returns:
        CompiledGate: The compiled gate.
        List[tuple] (optional): The algorithm's history. It is a list
            of tuples, each of form
                (recursion_depth, error_history, circuit_sizes)
            Only returned if `return_history` is set to True.
    '''
    if scope == 'full':
        # Deep copies to avoid pointers
        search_list = [i for i in net.net]
    elif scope == 'last_layer':
        search_list = [i for i in net._cur_layer]
    else:
        raise ValueError(
            '`scope` needs to be either "full" or "last_layer".'
        )

    if shuffle:
        random.shuffle(search_list)

    if not (bucket_params is None):
        k = bucket_params['k']
        bucket_size = bucket_params['bucket_size']
        bucket_robustness = bucket_params['bucket_robustness']
        bucket = Bucket(search_list=search_list,
                        k=k,
                        bucket_size=bucket_size)
    else:
        bucket = None
        bucket_robustness = 0

    if verbosity:
        print('Total search size: ', len(search_list))

    hist = []
    sk_result = solovay_kitaev_dev(
        gate=gate,
        depth=depth,
        search_list=search_list,
        epsilon_0=epsilon_0,
        scale=scale,
        method=method,
        miim_method=miim_method,
        bucket=bucket,
        bucket_robustness=bucket_robustness,
        verbosity=verbosity,
        _hist=hist
    )

    # Last depth history
    error = frobenius_norm(sk_result.total_unitary, gate)
    len_prev = len(sk_result)
    hist.append((depth, error, len_prev))

    if return_history:
        return sk_result, hist
    else:
        return sk_result
