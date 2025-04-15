# ECS130 HW2 -- QR Factorization Algorithms
import linear_systems
import numpy as np
from numpy.linalg import norm

def modified_gram_schmidt(A):
    """ Construct a reduced QR decomposition of `A` using the modified Gram-Schmidt algorithm. """
    m, n = A.shape
    if (m < n): raise ValueError("A must have at least as many rows as columns")

    Q = np.empty((m, n))
    R = np.empty((n, n))

    # TODO (Problem 9): Implement the modified Gram-Schmidt algorithm
    # https://julianpanetta.com/teaching/ECS130/10-QR-part2-deck.html#/modified-gram-schmidt-algorithm/5
    
    return Q, R

def householder(A):
    """
    Construct a full QR decomposition of `A` using the Householder algorithm.
    The Q factor is represented in implicit form as a list of reflectors "v".
    """
    m, n = A.shape
    if (m < n): raise ValueError("A must have at least as many rows as columns")

    # TODO (Problem 11): Implement the Householder algorithm
    # https://julianpanetta.com/teaching/ECS130/10-QR-part2-deck.html#/householder-qr-algorithm/9
    Q = []
    R = A.copy()

    return Q, R

def apply_householder_Q_transpose(Q, b):
    """ Compute the result of Q^T B using the same representation of Q returned
    by your `householder` function. """
    # TODO (Problem 12)
    y = b.copy()
    return y

# The different methods for solving a least-squares problem $A x = b$
# and their corresponding factorization algorithms.
methods = {
    'ModifiedGramSchmidt':         modified_gram_schmidt,
    'Householder':                 householder,
    'ModifiedGramSchmidtImproved': modified_gram_schmidt
}

def least_squares(A, b, method):
    """ Solves "A x = b" using the QR decomposition approach """

    if method not in methods: raise ValueError(f'Unknown method: {method}')
    factorizer = methods[method]

    m, n = A.shape
    Q, R = factorizer(A)
    result = np.empty(A.shape[1])
    result[:] = np.nan # NaN signals to the caller that the method is unimplemented

    if method == 'ModifiedGramSchmidt':
        # TODO (Problem 10): Implement the least-squares solution using the
        # reduced QR decomposition from the modified Gram-Schmidt algorithm.
        pass
    elif method == 'Householder':
        # TODO (Problem 12): Implement the least-squares solution using the
        # full QR decomposition from the Householder algorithm.
        pass
    elif method == 'ModifiedGramSchmidtImproved':
        pass

    return result

from relerror_unittest import *
class TestCases(RelerrorTestCase):
    def test_modified_gram_schmidt(self):
        # Verify that the MGS implementation produces a valid reduced QR factorization (Problem 9)
        for i in range(100):
            m = np.random.randint(1, 100)
            n = np.random.randint(1, 100)
            if m < n: m, n = n, m # Only consider tall matrices
            A = np.random.normal(size=(m, n))
            A_orig = A.copy()

            Q, R = modified_gram_schmidt(A)
            self.requireSame(A, A_orig, tol=0, msg='modified_gram_schmidt should not modify A')

            self.assertTrue(Q.shape == (m, n) and R.shape == (n, n), msg='Q and/or R have incorrect dimensions')

            R[np.tril_indices(n, -1)] = 0 # Zero out the subdiagonal of R (in case it was filled with garbage)

            self.requireSame(Q @ R, A_orig,             msg='A != QR') # Check that backward error is small
            self.requireSame(Q.T @ Q, np.eye(n),        msg="Q's columns are not orthonormal")
            self.assertTrue(np.all(np.diagonal(R) > 0), msg='R has nonpositive diagonal entries')

    def test_householder(self):
        # Verify that the Householder implementation produces a valid full QR factorization
        # with Q represented implicitly (Problems 11 and 12)
        def run_householder_test(A):
            A_orig = A.copy()

            Q_rep, R = householder(A)
            self.requireSame(A, A_orig, tol=0, msg='householder should not modify A')

            self.assertTrue(R.shape == (m, n), msg='R has incorrect dimensions')

            R[np.tril_indices(n, -1)] = 0 # Zero out the subdiagonal of R (in case it was filled with garbage)

            # Build an explicit representation of Q from the reflectors returned by `householder`
            Q_t = np.eye(m)
            for i in range(m):
                Q_t[:, i] = apply_householder_Q_transpose(Q_rep, Q_t[:, i])
            Q = Q_t.T

            self.requireSame(Q.T @ Q, np.eye(m), msg="Q's columns are not orthonormal")
            self.requireSame(Q @ R, A_orig,      msg='A != QR') # Check that backward error is small

        for i in range(100):
            m = np.random.randint(1, 100)
            n = np.random.randint(1, 100)
            if m < n: m, n = n, m # Only consider tall matrices

            A = np.random.normal(size=(m, n))
            run_householder_test(A)

            # Test with a zero diagonal: this is needed to verify `sign` is computed according to
            # slides' definitions (and not just using `np.sign` which returns 0 for 0).
            np.fill_diagonal(A, 0)
            run_householder_test(A)

            # Test with a zero matrix:
            # We skip this because handling it properly requires special handling for the `norm(v_k) == 0` case,
            # which was not included in the original pseudocode.
            # run_householder_test(np.zeros((m, n)))

if __name__ == '__main__':
    # Run the tests defined above.
    np.random.seed(0)
    unittest.main()
