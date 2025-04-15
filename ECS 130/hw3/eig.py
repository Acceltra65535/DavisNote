# ECS 130 HW3 -- Eigenvalue Decomposition via the QR Algorithm
import qr as qr
import numpy as np
from numpy.linalg import norm

def right_multiply_Q(B, v_list):
    """
    Replace the contents of `B` with the product `B Q`, where orthogonal matrix
    `Q` is represented implicitly by the list of Householder vectors `v_list`.
    """
    # TODO (Problem 2a): apply each Householder reflector in `v_list` to each *row* of `B`
    pass

def qr_iteration(A, Q_accum):
    """
    Apply a single iteration of the QR Eigenvalue algorithm to symmetric
    matrix `A`, accumulating the iteration's Q factor to `Q_accum`
    """
    # TODO (Problem 2b): update A and Q_accum in-place!
    pass

def off_diag_size(A):
    """
    Compute the norm of the off-diagonal elements of a square matrix `A`.
    """
    return np.sqrt(2) * norm(A[np.triu_indices(A.shape[0], 1)])

def pure_qr(A, Q_accum, tol=1e-8, maxIterations=1000):
    """
    Run the simplest, barebones implementation of the QR algorithm
    (without shifts or deflation) to reduce `A` to a diagonal matrix
    via an orthogonal similarity transform that is multiplied into
    `Q_accum`.
    Iteration is terminated when the off-diagonal's relative magnitude
    shrinks below `tol` or when `maxIterations` iterations have been run
    (whichever comes first).
    """
    # Use the householder QR algorithm to compute the eigenvalues
    # and eigenvectors of the symmetric matrix A
    residuals = []
    for i in range(maxIterations):
        qr_iteration(A, Q_accum)
        odiag = off_diag_size(A)
        residuals.append(odiag)
        if odiag < tol:
            break
    return residuals

def full_qr(A, Q_accum, tol=1e-8):
    # TODO (Problem 3): Implement the QR algorithm with the Rayleigh quotient shift and deflation.
    # Also record the residuals at each step in `residuals` like done in `pure_qr` above.
    residuals = []

    return residuals

def sorted_eigendecomposition(A, tol=1e-8, descending=True):
    """
    Compute the eigenvalue decomposition using `full_qr` and then sort
    the eigenvalues/permute the eigenvectors so that the diagonal of `A`
    is descending (like in the SVD) or ascending.
    """
    A = A.copy()
    m = A.shape[0]
    Q = np.eye(m)
    residuals = full_qr(A, Q, tol)
    p = np.argsort(np.diag(A))
    if descending: p = p[::-1]
    return np.diag(A)[p], Q[:, p]

import unittest
from matplotlib import pyplot as plt
import sys

from relerror_unittest import *
class TestCases(RelerrorTestCase):
    def test_right_multiply_Q(self):
        for B_in, v_list_in, B_out in TestData('right_multiply_Q').data:
            right_multiply_Q(B_in, v_list_in)
            self.requireSame(B_in, B_out, msg='Incorrectly updated B matrix from right_multiply_Q')

    def test_qr_iteration(self):
        for A_in, Q_accum_in, A_out, Q_accum_out in TestData('qr_iteration').data:
            qr_iteration(A_in, Q_accum_in)
            self.requireSame(A_in,       A_out,       msg='Incorrectly updated A matrix from qr_iteration')
            self.requireSame(Q_accum_in, Q_accum_out, msg='Incorrectly updated Q_accum matrix from qr_iteration')

    def test_full_qr(self):
        for A_in, Q_accum_in, A_true, Q_accum_true, residuals_true in TestData('full_qr').data:
            residuals = full_qr(A_in, Q_accum_in)
            self.requireSame(A_in,                A_true,                   msg='Incorrect diagonal factor (updated A) from full_qr')
            self.requireSame(Q_accum_in,          Q_accum_true,             msg='Incorrect eigenvectors (updated Q_accum) from full_qr')
            self.requireSame(np.array(residuals), np.array(residuals_true), msg='Incorrect per-iteration residuals from full_qr')

if __name__ == '__main__':
    if '--test' in sys.argv or '-t' in sys.argv:
        # Run the unit tests defined above.
        unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=1)
        sys.argv = [a for a in sys.argv if a not in ['-t', '--test']]
    m = int(sys.argv[1]) if len(sys.argv) == 2 else 5

    # Generate a random symmetric matrix.
    A = np.random.normal(size=(m, m))
    A += A.T
    Q = np.eye(m)
    pure_qr_lambda = A.copy()
    pure_qr_Q = Q.copy()
    residuals_pure = pure_qr(pure_qr_lambda, pure_qr_Q)
    print(f'Computing eigendecomposition of a random symmetric {m}x{m} matrix...')
    print(f'Pure QR off-diagonal magnitude:\t{residuals_pure[-1]}')
    print(f'Pure QR reconstruction error:\t{norm(A - pure_qr_Q @ pure_qr_lambda @ pure_qr_Q.T)}')
    plt.semilogy(residuals_pure, label='pure_qr')
    plt.legend()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Off-diagonal magnitude')
    plt.title(f'QR Algorithm Convergence for a Random {m}x{m} Matrix')
    plt.savefig('residuals.pdf')
    plt.close()

    𝜦 = A.copy()
    residuals_full = full_qr(𝜦, Q)

    if len(residuals_full) == 0: print('No residuals returned from full_qr; exiting early'); sys.exit(1)

    print(f'Full QR off-diagonal magnitude:\t{residuals_full[-1]}')
    print(f'Full QR reconstruction error:\t{norm(A - Q @ 𝜦 @ Q.T)}')

    np.set_printoptions(edgeitems=100, linewidth=1000)

    plt.semilogy(residuals_pure, label='pure_qr')
    plt.semilogy(residuals_full, label='full_qr')
    plt.xlabel('Iteration')
    plt.ylabel('Off-diagonal magnitude')
    plt.title(f'QR Algorithm Convergence for a Random {m}x{m} Matrix')
    plt.legend()
    plt.grid()
    plt.savefig('residuals_full_qr.pdf')
