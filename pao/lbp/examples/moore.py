#
# Example from
# Moore, J. and J. Bard 1990.
# The mixed integer linear bilevel programming problem.
# Operations Research 38(5), 911–921.
#
from pao.lbp import *


def create():
    M = LinearBilevelProblem()

    U = M.add_upper(nxZ=1)
    U.c.U.x = [-1]
    U.c.L.x = [-10]

    L = M.add_lower(nxZ=1)
    L.c.L.x = [1]

    L.A.U.x = [[-25],
               [1],
               [2],
               [-2]]
    L.A.L.x = [[20],
               [2],
               [-1],
               [-10]]
    L.b = [30,10,15,-15]

    return M


if __name__ == "__main__":          #pragma: no cover
    M = create()
    opt = SolverFactory('pao.lbp.FA')
    opt.solve(M)
    M.print()
