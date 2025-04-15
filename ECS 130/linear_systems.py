# ECS130 HW1 -- Solving Linear Systems
import numpy as np, scipy
import matplotlib

def lu(A):
    """ Construct an LU decomposition of `A` without pivoting. """
    m = A.shape[0]
    if A.shape[1] != m: raise Exception('A is not square!')
    L, U = np.identity(m), A.copy()

    for k in range(m - 1):
        for i in range(k + 1, m):
            L[i, k] = U[i, k] / U[k, k]

            for j in range(k + 1, m):
                U[i, j] -= L[i, k] * U[k, j]
            U[i, k] = 0

    return L, U

def forwardsub(L, b):
    """ Solve "L x = b" with a lower triangular matrix L. """
    m = L.shape[0]
    if L.shape[1] != m: raise Exception('L is not square!')
    if b.shape != (m,): raise Exception('b is not a correctly sized vector!')

    x = b.copy()
    for i in range(m):
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]

    return x

def backsub(U, b):
    """ Solve "U x = b" with an upper triangular matrix U. """
    m = U.shape[0]
    if U.shape[1] != m: raise Exception('U is not square!')
    if b.shape != (m,): raise Exception('b is not a correctly sized vector!')

    x = b.copy()
    for i in range(m - 1, -1, -1):
        if U[i, i] == 0:
            return 0
        x[i] /= U[i, i]
        for j in range(i):
            x[j] -= U[j, i] * x[i]

    return x

def solve(A, b):
    """ Solve "A x = b" using the LU decomposition approach """
    m = A.shape[0]
    U = A.copy()
    L = np.identity(m) #create matrix

    for k in range(m-1):
        for i in range(k+1, m):
            if U[k, k] == 0:
                return 0
            fa = U[i, k] / U[k, k]
            L[i, k] = fa
            U[i, k:] -= fa * U[k, k:]
    
    y = forwardsub(L, b)
    x = backsub(U, y)
    b[:] = x

    return b

def forwardsub_vectorized(L, b):
    """ Solves "L x = b" with a lower triangular matrix L. """
    m = L.shape[0]
    if L.shape[1] != m:raise Exception('L is not square!')
    if b.shape != (m,):raise Exception('b is not a correctly sized vector!')
    
    x = b.copy()
    for i in range(m):
        x[i] -= np.dot(L[i, :i], x[:i])
        x[i] /= L[i, i]

    return x

def relerror(a, b):
    """ Calculate the relative difference between vectors/matrices `a` and `b`. """
    return np.linalg.norm(a - b) / np.linalg.norm(b)

def syntheticFactors(m, order='F'):
    """ Generate a random unit lower triangular matrix L and upper triangular matrix U. """
    L = np.array(np.random.uniform(size=(m, m)), order=order)
    U = np.array(np.random.uniform(size=(m, m)), order=order)
    L[np.triu_indices(m, 1)] = 0
    L[np.diag_indices(m)] = 1
    U[np.tril_indices(m, -1)] = 0
    return L, U

def forwardsub_scipy(L, b):
    return scipy.linalg.solve_triangular(L, b, lower=True, check_finite=False)

forwardsub_implementations = {'loop': forwardsub,
                              'vectorized': forwardsub_vectorized,
                              'scipy': forwardsub_scipy}

import unittest
class TestCases(unittest.TestCase):
    def requireSame(self, a, b, tol = 1e-10, msg=None):
        self.assertLessEqual(relerror(a, b), tol, msg=msg)

    def test_lu_slides_example(self):
        A = np.array([[2, 1, 1, 0],
                      [4, 3, 3, 1],
                      [8, 7, 9, 5],
                      [6, 7, 9, 8]], dtype=float)
        L, U = lu(A)
        self.requireSame(np.tril(L), np.array([[1., 0., 0., 0.], [2., 1., 0., 0.], [4., 3., 1., 0.], [3., 4., 1., 1.]]))
        self.requireSame(np.triu(U), np.array([[2., 1., 1., 0.], [0., 1., 1., 1.], [0., 0., 2., 2.], [0., 0., 0., 2.]]))

    def test_lu_synthetic(self):
        for i in range(100):
            m = np.random.randint(1, 10)
            L, U = syntheticFactors(m)
            Lc, Uc = lu(L @ U)
            self.requireSame(np.tril(Lc), L, msg=f'Error in L factor')
            self.requireSame(np.triu(Uc), U, msg=f'Error in U factor')

    def test_forwardsub_synthetic(self):
        for i in range(100):
            m = np.random.randint(1, 10)
            L, U = syntheticFactors(m)
            x = np.random.normal(size=m)
            b = L @ x
            b_orig = b.copy()
            for name, fs in forwardsub_implementations.items():
                try:
                    self.requireSame(x, fs(L, b), msg=f'Error in forwardsub implementation "{name}"')
                    self.requireSame(b, b_orig, tol=0, msg=f'forwardsub implementation "{name}" should not modify b!')
                except NotImplementedError: # Skip the vectorized implementation if it's not implemented
                    pass

    def test_backsub_synthetic(self):
        for i in range(100):
            m = np.random.randint(1, 10)
            L, U = syntheticFactors(m)
            x = np.random.normal(size=m)
            b = U @ x
            b_orig = b.copy()
            self.requireSame(x, backsub(U, b), msg=f'Error in backsub')
            self.requireSame(b, b_orig, tol=0, msg='backsub should not modify b!')

    def test_solve_synthetic(self):
        for i in range(100):
            m = np.random.randint(1, 10)
            A = np.random.uniform(size=(m, m))
            x = np.random.normal(size=m)
            b = A @ x
            self.requireSame(x, solve(A, b), msg=f'Error in solve')

    def test_forwardsub_vectorized_synthetic(self):
        for i in range(100):
            m = np.random.randint(1, 10)
            L, _ = syntheticFactors(m)
            x = np.random.normal(size=m)
            b = L @ x
            b_orig = b.copy()
            self.requireSame(x, forwardsub_vectorized(L, b), msg=f'Error in implementation')
            self.requireSame(b, b_orig, tol=0, msg=f'forwardsub_vectorized implementation should not modify b!')
            
if __name__ == '__main__':
    import sys
    if '-b' in sys.argv:
        # Benchmarking
        from matplotlib import pyplot as plt
        import timeit

        setup_command = lambda m, impl: f'from __main__ import syntheticFactors, forwardsub_implementations; import numpy as np; L, U = syntheticFactors({m}); b = np.random.normal(size={m}); forwardsub = forwardsub_implementations["{impl}"]'
        times = {}
        for impl in forwardsub_implementations:
            try:
                sizes = [16, 32, 64, 128, 200, 256, 400, 512, 756, 1024]
                if impl != 'loop': # Include larger matrices for the faster variants
                    sizes.extend([1600, 2048, 3000, 4096])
                time = [min(timeit.repeat(f'forwardsub(L, b)', setup_command(m, impl), number=10, repeat=2)) for m in sizes]
                plt.loglog(sizes, time,  '-s', ms=4, label=impl)
            except NotImplementedError: pass # Skip the vectorized implementation if it's not implemented

        plt.xlabel('Matrix size $m$')
        plt.ylabel('Time per substitution (s)')
        plt.title('Forward Substitution Benchmarks')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('forwardsub_benchmarks.pdf')
        plt.close()

    # Run the tests defined above.
    np.random.seed(0)
    unittest.main()
