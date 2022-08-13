""" Dedalus-nsolver interface class for specifying dynamical systems in dedalus

    The setting up PDEs in dedalus for timestepping, computing observables (norms),
    updating parameters and communications with nsolver library in C++ are specified
    here

"""

from abc import ABC, abstractmethod 

class DedalusInterface(ABC):

    def __init__(self):
        print("-----------dedalus problem setup complete--------------\n")

    def domain_setup(self):
        """Setup dedalus domain/grid.
        
        Can accept arguments to update domain size during continuations
        """
        pass 

    def problem_setup(self):
        """Setup the variables, equations and boundary conditions.
        
        Can also accept paramaters to update the system during continuations
        """
        pass 

    def build_solver(self):
        """Build the dedalus solver and set timesteppers, simulation time/number of iterations"""
        pass 
    
    def init_problem(self):
        """Intialize the system in dedalus"""
        pass 

    def shifts(self):
        """Shift operations on system state for special solutions."""
        pass 

    @abstractmethod
    def add_perturbations(self, mag, decay):
        """Create perturbation fields in order to compute eigenvalue spectrum.
        
        Returns perturbation as nsolver vector of the same size as field variables
        It must be verified that the perturbed field satisfies additional constraints and
        boundary conditions 
        Keyword arguments:
        mag   -> perturbation magnitude, given as 2-norm of the field
        decay -> decay in spectral perturbations for larger wavenumbers (default = 1)
        """
        pass 
        
    @abstractmethod
    def observable(self, x):
        """Observable of the system state, used for performing continuation.
        
        Returns a scalar of interest, such as L2norm, dissipation, energy, etc.
        Keyword arguments:
        x -> nsolver vector 
        """
        pass 
        
    @abstractmethod
    def diff(self, x, dir):
        """Compute partial derivatives of the fields in a given direction.

        Can be employed to satisfy orthogonality constraints in nsolver for
        for additional unknowns such as speeds of travelling solutions.
        Returns back an nsolver vector of same length as x 
        Keyword arguments:
        x   -> nsolver vector
        dir -> axis for partial differentiation, currently 'x' and 'y' are 
        the only valid arguments
        """
        pass 

    @abstractmethod
    def advance(self, T, x):
        """Time integration of the system from given initial state
        
        Used by nsolver to approximate the Jacobians by finite differences
        Returns the evolved state as an nsolver vector with same size as x 
        Keyword argments:
        T -> time of integration
        x -> nsolver vector
        """
        pass 
        
    @abstractmethod
    def updateMu(self, mu, muName):
        """Update the contiuation parameter and reconstruct dedalus system.
        
        For updating system parameters, dedalus solver must be reconstructed at each
        continuation step
        In addition, if continuation in domain size is desired, the dedalus domain
        must be reconstructed along with the solver for each continuation step

        Keyword argments
        mu     -> parameter value
        muName -> parameter name, given as a string 
        """
        pass 
    
    @abstractmethod
    def read_h5(self, filename, iter):
        """Interface to read from h5 files in nsolver.

        Returns an nsolver vector with the state loaded from h5 file
        Keyword arguments:
        filename -> name of the h5 file, given as a string
        iter     -> iteration number
        """
        pass 
        
    def symmetry(self):
        """Perform symmetry operations on the state ."""
        pass 
        