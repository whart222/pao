#
# Solvers that convert the problem to a LinearBilevelProblem and
# solve using a LBP-specific solver.
#
import time
import pao.lbp
import pao.common
from pao.bilevel.solver import SolverFactory, PyomoSubmodelSolverBase_LBP


@SolverFactory.register(
        name='pao.submodel.FA',
        doc=SolverFactory.doc('pao.lbp.FA'))
class PyomoSubmodelSolver_FA(PyomoSubmodelSolverBase_LBP):

    def __init__(self, **kwds):
        super().__init__('pao.submodel.FA', 'pao.lbp.FA', False)

@SolverFactory.register(
        name='pao.submodel.REG',
        doc=SolverFactory.doc('pao.lbp.REG'))
class PyomoSubmodelSolver_REG(PyomoSubmodelSolverBase_LBP):

    def __init__(self, **kwds):
        super().__init__('pao.submodel.REG', 'pao.lbp.REG', False)

@SolverFactory.register(
        name='pao.submodel.PCCG',
        doc=SolverFactory.doc('pao.lbp.PCCG'))
class PyomoSubmodelSolver_PCCG(PyomoSubmodelSolverBase_LBP):

    def __init__(self, **kwds):
        super().__init__('pao.submodel.PCCG', 'pao.lbp.PCCG', False)
        self.config.epsilon = 1e-4    #For use in disjunction approximation
        self.config.xi = 0                #tolerance for UB-LB to claim convergence
        self.config.maxit = 5           #Maximum number of iterations
        self.config.bigm = 1e6             #upper bound on variables


