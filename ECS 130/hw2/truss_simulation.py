# ECS 130 HW2: Truss Simulation
import visualization
from matplotlib import pyplot as plt
import numpy as np
import pickle
import linear_systems

def construct_B(X, E):
    """
    Construct the B matrix for a truss structure given the rest joint positions
    in the rows of X and edge endpoint indices in the rows of E.
    """
    # TODO (Problem 14)
    return np.array([])

def construct_K(X, E, k):
    """
    Construct the stiffness matrix K for a truss structure given the rest joint
    positions in the rows of X, edge endpoint indices in the rows of E, and the
    stiffness k of each edge.
    """
    numNodes = X.shape[0] # referred to as n in the handout
    numEdges = E.shape[0] # referred to as m in the handout

    # TODO (Problem 14): Construct the stiffness matrix K

    return np.identity(2 * numNodes) # (replace this...)

def apply_bcs(C, K, F):
    # TODO (Problem 15): modify K and F *in pace* to apply constraints `u_c = 0`
    # for each node `c` in `C` (converting them to the modified system
    # matrix and right-hand side K_tilde and F_tilde described in the handout).
    pass # (replace this...)

def simulate(X, E, C, F, k):
    """
    Simulates the deformation of a truss structure given the rest joint positions
    in the rows of X, edge endpoint indices in the rows of E, indices of fixed
    joints in the array C, and the stiffness k of each edge.
    """

    # TODO (Problem 15): Simulate the deformation of the truss structure

    # (Replace the following...)
    numNodes = X.shape[0] # referred to as n in the handout
    return np.zeros((2 * numNodes,))

from relerror_unittest import *
class TestCases(RelerrorTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = pickle.load(open('data/simulation_test_data.pkl', 'rb'))

        from collections import namedtuple
        STC = namedtuple('SimTestCase', ['X', 'E', 'F', 'C', 'k', 'B', 'K', 'K_tilde', 'F_tilde', 'U'])
        for name, data in self.data.items():
            self.data[name] = STC(**data)

    def test_construct_B(self):
        for name, data in self.data.items():
            B = construct_B(data.X, data.E)
            self.assertTrue(B.shape == data.B.shape, msg='Incorrect B shape')
            self.requireSame(B, data.B, msg='Incorrect B matrix')

    def test_construct_K(self):
        for name, data in self.data.items():
            self.requireSame(construct_K(data.X, data.E, data.k), data.K, msg='Incorrect K matrix')

    def test_apply_bcs(self):
        for name, data in self.data.items():
            K_tilde = data.K.copy()
            F_tilde = data.F.copy().ravel()
            apply_bcs(data.C, K_tilde, F_tilde)
            self.requireSame(K_tilde, data.K_tilde, msg='Incorrect K_tilde matrix (apply_bcs)')
            self.requireSame(F_tilde, data.F_tilde, msg='Incorrect F_tilde vector (apply_bcs)')

    def test_simulate(self):
        for name, data in self.data.items():
            self.requireSame(simulate(data.X, data.E, data.C, data.F.ravel(), data.k), data.U, msg='Incorrect simulation result')

if __name__ == '__main__':
    import sys

    if ('-t' in sys.argv) or ('--test' in sys.argv):
        # Run the tests defined above.
        np.random.seed(0)
        unittest.main(argv=[sys.argv[0]])

        # Quit if only the `--test` flag is present; otherwise
        # remove the flag and try to run the simulation.
        sys.argv = [arg for arg in sys.argv if arg not in ['-t', '--test']]
        if (len(sys.argv) == 1): sys.exit()

    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 truss_simulation.py <data file>")
        print("or:    python3 truss_simulation.py --test")
        sys.exit(1)

    data = pickle.load(open(sys.argv[1], 'rb'))

    name = data['name']
    X = data['X']
    E = data['E']
    C = np.array(data['C'])
    F = data['F']
    k = data['k']

    # Visualize the truss's rest configuration
    axisLimits = visualization.visualizeTruss(X, E, C, F)
    plt.savefig(f'{name}_init.pdf')
    plt.close()

    # Simulate the truss's deformation
    U = simulate(X, E, C, F.ravel(), k).reshape((-1, 2))
    x = X + U

    # Visualize the truss's deformed configuration
    visualization.visualizeTruss(x, E, C, F, axisLimits=axisLimits)
    plt.savefig(f'{name}_deformed.pdf')
    plt.close()
