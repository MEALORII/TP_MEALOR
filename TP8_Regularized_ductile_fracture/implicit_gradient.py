#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Convenience class for solving implicit gradient variables

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   25/07/2023
"""

import ufl
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem
from dolfinx_materials.utils import project

class ImplicitGradient:
    def __init__(self, domain, ell, degree=1, axisymmetrical = False):
        """Implicit gradient class for scalar variables

        Parameters
        ----------
        domain : Mesh
            A dolfinx mesh
        ell : float, fem.Constant
            Internal length scale
        degree : int
            Polynomial degree of function space
        """
        self.domain = domain
        self.ell = ell
        self.V = fem.FunctionSpace(self.domain, ("CG", degree))
        self.axisymmetrical = axisymmetrical
        
        # Define variational problem for projection
        self.chi_ = ufl.TestFunction(self.V)
        self.dchi = ufl.TrialFunction(self.V)
        self.dx = ufl.Measure("dx", domain=domain)
        
        if not self.axisymmetrical:
            a = fem.form(
                ufl.inner(self.dchi, self.chi_) * self.dx
                + self.ell**2
                * ufl.inner(ufl.grad(self.dchi), ufl.grad(self.chi_))
                * self.dx
            )
        else:
            def axi_grad(v):
                return ufl.as_vector([v.dx(0), 0, v.dx(1)])
            x = ufl.SpatialCoordinate(self.domain)
            a = fem.form(
                ufl.inner(self.dchi, self.chi_) * x[0] * self.dx
                + self.ell**2
                * ufl.inner(axi_grad(self.dchi), axi_grad(self.chi_))
                * x[0] * self.dx
            )
        # Assemble linear system
        A = fem.petsc.assemble_matrix(a)
        A.assemble()

        # Setup linear solver
        self.solver = PETSc.KSP().create(A.getComm())
        self.solver.setOperators(A)
        self.solver.setType(PETSc.KSP.Type.CG)
        pc = self.solver.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)

    def smooth(self, chi, chi_smoothed):
        Vchi = chi.function_space.ufl_element()
        if Vchi.family() == "Quadrature":
            degree = Vchi.degree()
            dx = self.dx(metadata={"quadrature_degree": degree})
        else:
            dx = self.dx

        if not self.axisymmetrical:
            L = fem.form(ufl.inner(chi, self.chi_) * dx)
        else:
            x = ufl.SpatialCoordinate(self.domain)
            L = fem.form(ufl.inner(chi, self.chi_) * x[0] * dx)
        b = fem.petsc.assemble_vector(L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.solver.solve(b, chi_smoothed.vector)
        chi_smoothed.x.scatter_forward()
        return chi_smoothed
