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
    def __init__(self, domain, ell, degree=1):
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

        # Define variational problem for projection
        self.chi_ = ufl.TestFunction(self.V)
        self.dchi = ufl.TrialFunction(self.V)
        self.dx = ufl.Measure("dx", domain=domain)
        a = fem.form(
            ufl.inner(self.dchi, self.chi_) * self.dx
            + self.ell**2
            * ufl.inner(ufl.grad(self.dchi), ufl.grad(self.chi_))
            * self.dx
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

        L = fem.form(ufl.inner(chi, self.chi_) * dx)
        b = fem.petsc.assemble_vector(L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.solver.solve(b, chi_smoothed.vector)
        chi_smoothed.x.scatter_forward()
        return chi_smoothed


class ImplicitGradient2:
    def __init__(self, domain, ell, degree=1):
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
        self.chi_s = fem.Function(self.V, name="Smoothed")
        self.chi_ = ufl.TestFunction(self.V)
        dchi = ufl.TrialFunction(self.V)
        self.dx = ufl.Measure("dx", domain=domain)
        self.a = (
            ufl.inner(self.chi_, dchi)
            + self.ell**2 * ufl.inner(ufl.grad(self.chi_), ufl.grad(dchi))
        ) * self.dx
        self.A = fem.petsc.create_matrix(fem.form(self.a))

        L = self.chi_ * self.dx
        self.b = fem.petsc.create_vector(fem.form(L))
        fem.petsc.assemble_matrix(self.A, fem.form(self.a))
        self.A.assemble()

        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setOperators(self.A)
        # self.solver.setType(PETSc.KSP.Type.CG)
        # pc = self.solver.getPC()
        # pc.setType(PETSc.PC.Type.HYPRE)

    def smooth(self, chi, chi_s=None):
        if chi_s is None:
            chi_s = self.chi_s
        # chi_s.vector.array[:] = 0.0
        L = ufl.inner(chi, self.chi_) * self.dx
        fem.petsc.assemble_vector(self.b, fem.form(L))
        self.solver.solve(self.b, chi_s.vector)
        chi_s.x.scatter_forward()
        print(chi_s.vector.array[:])
        # project(chi, chi_s, smooth=self.ell)
        print(chi_s.vector.array[:])
        return chi_s
