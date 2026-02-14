'''
------------------------------------------------------------------------------
Author: Henrique Ennes (https://hlovisiennes.github.io/)
------------------------------------------------------------------------------
Utility functions.
------------------------------------------------------------------------------
Utils:
    frobenius_norm
    commutator
    Rx
    Rz
    su_matrix
    su2_matrix
    lift
    gell_mann_su3
    test_inverse
    test_unitarity
    test_special_unitarity
    progress_report
    progress_bar
    plot_history
------------------------------------------------------------------------------
'''
from typing import List, Optional
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.stats import unitary_group

'''
Linear algebra operations
'''


def frobenius_norm(
        matrix1: np.ndarray,
        matrix2: np.ndarray,
        centralizer: bool = False
        ) -> float:
    '''
    Computes the Frobenius norm distance between
    two general matrices.

    Args:
        matrix1(np.ndarray): First matrix.
        matrix2(np.ndarray): Second matrix.
        centralizer (bool): Whether to take metric with respect to the
            centralizer of SU(d) (i.e., in practice, we compare only
            matrices up to PSU(d)). In practice, this is done
            by taking min_i(||U-w^i V||), where w = exp(2 pi i/d).
            Defaults to True.

    Returns:
        float: The Frobenius norm distance.
    '''
    n = matrix1.shape[0]

    if centralizer:
        return min([
            frobenius_norm(matrix1,
                           matrix1*np.exp(1j*2*np.pi*i/n),
                           centralizer=False) for i in range(n)])
    else:
        return np.linalg.norm(matrix1 - matrix2)


def commutator(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    '''
    Computes the commutator of two matrices.

    Args:
        matrix1(np.ndarray): First matrix.
        matrix2(np.ndarray): Second matrix.

    Returns:
        np.ndarray: The commutator [A, B] = AB - BA.
    '''
    return matrix1 @ matrix2 - matrix2 @ matrix1


'''
Gate factory
'''


def Rx(theta: float) -> np.ndarray:
    '''
    Makes an rotation matrix along the X-axis of
    the Bloch sphere.

    Args:
        theta (float): Rotation angle.

    Returns:
        np.ndarray: Rotation matrix.
    '''
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ])


def Rz(theta):
    '''
    Makes an rotation matrix along the Z-axis of
    the Bloch sphere.

    Args:
        theta (float): Rotation angle.

    Returns:
        np.ndarray: Rotation matrix.
    '''
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ])


def su_matrix(dimension: int, seed: Optional[int] = None) -> np.ndarray:
    '''
    Generates a random SU(dimension) matrix by sampling in
    U(dimension) and projecting to SU(dimension).

    Args:
        dimension (int): The SU dimension.
        seed (int, optional): Seed for random number generating.
            Defaults to None.

    Returns:
        np.ndarray: Random SU matrix.
    '''
    if not (seed is None):
        rng = np.random.default_rng(seed=seed)
        U = unitary_group.rvs(dimension, random_state=rng)
    else:
        U = unitary_group.rvs(dimension)

    # Divide by the dimension root of the determinant to ensure
    # that matrix is special
    U /= np.linalg.det(U)**(1/dimension)
    return U


def gl_matrix(dimension: int, seed: Optional[int] = None) -> np.ndarray:
    '''
    Generates a random GL(dimension) Gaussian matrix centered at the
    identity and restricted to the unit sphere.

    Args:
        dimension (int): The GL dimension.
        seed (int, optional): Seed for random number generating.
            Defaults to None.

    Returns:
        np.ndarray: Random GL matrix.
    '''
    if not (seed is None):
        # Real
        rng = np.random.default_rng(seed=seed)
        real = rng.standard_normal((dimension, dimension)) / np.sqrt(2)

        # Imaginary
        rng = np.random.default_rng(seed=seed-1)
        imag = rng.standard_normal((dimension, dimension)) / np.sqrt(2)

        U = real + 1j * imag
    else:
        U = (np.random.randn(dimension, dimension)
             + 1j * np.random.randn(dimension, dimension))

    return U / np.linalg.norm(U, 'fro')


def su2_matrix(x: float, y: float, z: float) -> np.ndarray:
    '''
    Generates a SU(2) matrix by exponentiating an
    element of the Lie algebra.

    Args:
        x (float): The Pauli-X component.
        y (float): The Pauli-Y component.
        z (float): The Pauli-Z component.

    Returns:
        np.ndarray": Matrix in SU(2).
    '''
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    H = x * sx + y * sy + z * sz

    return expm(1j * H)


def lift(
        matrix: np.ndarray,
        lift_term: complex = np.exp(3j*np.pi/5)
        ) -> np.ndarray:
    '''
    Lifts a 2 x 2 matrix to 3 x 3 usingt the natural pronection
    on the first two coordinates.

    Args:
        matrix (np.ndarray): The 2 x 2 matrix to lift.
        lift_term (complex): The term to be set to the
            uncomputational basis.

    Returns:
        np.ndarray: A 3 x 3 matrix.
    '''
    lifted = np.eye(3).astype(complex)
    lifted[:2, :2] = matrix
    lifted[2, 2] = lift_term

    return lifted


def gell_mann_su3(
        x: float = np.pi/np.sqrt(2),
        y: float = np.pi/np.sqrt(5)
        ) -> tuple[np.ndarray]:
    '''
    Makes a universal set for SU(3) by exponentiating the Gell-Mann matrices.
    For this to be a generating set, we need thta x, y, z are irrationally
    related.

    Args:
        x (float): The lambda_1 component. Defaults to pi/sqrt(2).
        z (float): The lambda_4 component. Defaults to pi/sqrt(5).
    Returns:
        tuple[np.ndarray]: A universal set for SU(3).
    '''

    m1 = np.array([[np.cos(x), 1j*np.sin(x), 0],
                   [1j*np.sin(x), np.cos(x), 0],
                   [0, 0, 1]], dtype=complex)

    m2 = np.array([[np.cos(y), 0, 1j*np.sin(y)],
                   [0, 1, 0],
                   [1j*np.sin(y), 0, np.cos(y)]], dtype=complex)

    return m1, m2


'''
Some tests
'''


def test_inverse(
        matrix: np.ndarray,
        inverse: np.ndarray
        ) -> bool:
    '''
    Test whether a candidate inverse is indeed
    the inverse of some matrix.

    Args:
        matrix(np.ndarray): Matrix.
        inverse(np.ndarray): Candidate unitary.

    Returns:
        bool: Whether the inverse is indeed the inverse
            of the matrix (up to norm precision 1e-10).
    '''
    return frobenius_norm(
        matrix @ inverse, np.eye(matrix.shape[0])
        ) < 1e-10


def test_unitarity(matrix: np.ndarray) -> bool:
    '''
    Test whether a matrix is unitary.

    Args:
        matrix(np.ndarray): Matrix.

    Returns:
        bool: Whether the matrix is indeed unitary
            (up to norm precision 1e-10).
    '''

    return frobenius_norm(
        matrix.conj().T @ matrix, np.eye(matrix.shape[0])
        ) < 1e-10


def test_special_unitarity(matrix: np.ndarray) -> bool:
    '''
    Test whether a matrix is special unitary (i.e., unitary
    with determinant +1).

    Args:
        matrix(np.ndarray): Matrix.

    Returns:
        bool: Whether the matrix is indeed special unitary
            (up to norm precision 1e-10).
    '''
    return (
        test_unitarity(matrix)
        and np.abs(np.linalg.det(matrix) - 1) < 1e-10
        )


'''
Report functions
'''


def progress_report(current: int, total: int, tag: str):
    '''
    Shows progress of some evaluation.

    Args:
        current (int): How much of the process has been concluded.
        total (int): The process size.
        tag (str): The quantity being observed.

    Returns:
        None
    '''
    sys.stdout.write(f'\rProgress: {tag} {current} of {total}')
    sys.stdout.flush()

    if current == total:
        print()


def progress_bar(current: int, total: int):
    '''
    Makes progress bar for some evaluation.

    Args:
        current (int): How much of the process has been concluded.
        total (int): The process size.

    Returns:
        None
    '''
    bar_length = 30
    fraction = current / total
    filled_length = int(bar_length * fraction)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = fraction * 100
    sys.stdout.write(f'\r|{bar}| {percent:.1f}%')
    sys.stdout.flush()
    if current == total:
        print()


def plot_history(hist: List[tuple], save_plot: Optional[str] = None) -> None:
    '''
    Plots the relevant graphs relating Solovay-Kitaev's recursion depth, error,
    and circuit length.

    Args:
        list[tuple]: A Solovay-Kitaev history. Assumed
            a list of tuples, each of which
            (recursion_depth, error_history, circuit_sizes)
        save_plot (str, optional): The direction to save the plot. Defaults
            to None.
    '''
    hist = np.array(hist)

    errors = hist[:, 1]
    errors = np.log(errors)
    circuit_size = hist[:, 2]
    depth = hist[:, 0]

    _, axs = plt.subplots(1, 3, figsize=(30, 8))

    # Compilation length vs log error
    axs[0].scatter(circuit_size, errors)
    axs[0].set_xlabel('Compilation length', fontsize=15)
    axs[0].set_ylabel('log(error)', fontsize=15)

    # Recursion depth vs log Compilation length
    circuit_size = np.log(circuit_size)
    axs[1].scatter(depth, circuit_size)
    axs[1].set_xlabel('Depth', fontsize=15)
    axs[1].set_ylabel('log(compilation length)', fontsize=15)

    # Recursion depth vs log error
    axs[2].scatter(depth, errors)
    axs[2].set_xlabel('Depth', fontsize=15)
    axs[2].set_ylabel('log(errors)', fontsize=15)

    if save_plot:
        plt.savefig(save_plot)

    plt.show()
