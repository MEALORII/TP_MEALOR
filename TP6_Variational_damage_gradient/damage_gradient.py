#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:23:02 2019

@author: bleyerj
"""
import warnings
import numpy as np
from petsc4py import PETSc
from dolfinx import fem, la
import ufl
from ufl import (
    derivative,
    dot,
    grad,
    sym,
    inner,
    tr,
    dev,
    Identity,
)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mealor import DirichletBoundaryCondition
from mealor.utils import integrate, interpolate_expr, save_to_file
from mealor.tao_problem import TAOProblem
from IPython.display import clear_output


def solve_problem(domain, facets, prob_params, mech_params):
    problem_type, generate_bcs, bcs_damage, Nitermax, tol = prob_params
    E, nu, Gc, l0, model, Umax, Nincr = mech_params

    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facets)

    lmbda = fem.Constant(domain, E * nu / (1 + nu) / (1 - 2 * nu))
    mu = fem.Constant(domain, E / 2 / (1 + nu))
    lmbda_ps = (
        2 * mu * lmbda / (lmbda + 2 * mu)
    )  # effective lambda modulus for plane stress
    kres = fem.Constant(domain, 1e-6)  # residual stiffness
    l0 = fem.Constant(domain, l0)
    Gc = fem.Constant(domain, Gc)

    kappa = lmbda_ps + 2 / 3 * mu

    def ppos(u):
        return (u + abs(u)) / 2

    def pneg(u):
        return (-u + abs(u)) / 2

    def eps(u):
        return sym(grad(u))

    def sigma0(u):
        return lmbda_ps * tr(eps(u)) * Identity(2) + 2 * mu * eps(u)

    def g(d):
        return (1 - d) ** 2 + kres

    def psi(u):
        return 0.5 * inner(eps(u), sigma0(u))

    def sigma(u, d):
        return g(d) * sigma0(u)

    def psi_pos(u):
        ed = dev(eps(u))
        return 0.5 * kappa * ppos(tr(eps(u))) ** 2 + mu * inner(ed, ed)

    def psi_neg(u):
        return 0.5 * kappa * pneg(tr(eps(u))) ** 2

    load_steps = np.linspace(0, Umax, Nincr + 1)

    # function space for the displacement
    V_u = fem.VectorFunctionSpace(domain, ("CG", 1))
    # function space for the damage
    V_d = fem.FunctionSpace(domain, ("CG", 1))
    # function space for the stress
    V_sig = fem.TensorFunctionSpace(domain, ("DG", 0))

    Uimp = fem.Constant(domain, 1.0)
    dirichlet = DirichletBoundaryCondition(V_u)
    generate_bcs(dirichlet, Uimp)

    u = fem.Function(V_u, name="Total_displacement")
    v = ufl.TrialFunction(V_u)
    u_ = ufl.TestFunction(V_u)
    sig = fem.Function(V_sig, name="Current_stress")

    # Dummy function used to compute force reaction
    # fill with u=1 on imposed displacement boundary and use residual
    v_reac = fem.Function(V_u)
    fem.set_bc(v_reac.vector, dirichlet.bcs)

    d = fem.Function(V_d, name="Damage")
    dold = fem.Function(V_d, name="Previous_damage")
    dub = fem.Function(V_d, name="Upper_bound_d=1")
    with dub.vector.localForm() as bc_local:
        bc_local.set(1.0)
    dlb = fem.Function(V_d, name="Lower_bound_d_n")
    d_ = ufl.TestFunction(V_d)
    dd = ufl.TrialFunction(V_d)

    # total energy for the damage problem
    # elastic_energy = (g(d) * psi_pos(u) + psi_neg(u)) * dx
    elastic_energy = g(d) * psi(u) * dx

    if model == "AT1":
        cw = fem.Constant(domain, 8 / 3.0)
        w = lambda d: d
    elif model == "AT2":
        cw = fem.Constant(domain, 2.0)
        w = lambda d: d**2

    fracture_energy = Gc / cw * (w(d) / l0 + l0 * dot(grad(d), grad(d))) * dx
    total_energy = elastic_energy + fracture_energy

    F_u = derivative(elastic_energy, u, v)
    J_u = derivative(F_u, u, u_)

    problem_u = NonlinearProblem(
        F_u,
        u,
        bcs=dirichlet.bcs,
        J=J_u,
    )
    u_solver = NewtonSolver(domain.comm, problem_u)

    u_solver.convergence_criterion = "residual"
    u_solver.rtol = 1e-6
    u_solver.atol = 1e-6
    u_solver.max_it = 100
    u_solver.report = True

    if bcs_damage:
        left_dofs = fem.locate_dofs_topological(
            V_d, facets.dim, facets.indices[facets.values == 1]
        )
        right_dofs = fem.locate_dofs_topological(
            V_d, facets.dim, facets.indices[facets.values == 3]
        )
        bc_d = [
            fem.dirichletbc(0.0, left_dofs, V_d),
            fem.dirichletbc(0.0, right_dofs, V_d),
        ]
    else:
        bc_d = []

    # first derivative of energy with respect to d
    F_dam = derivative(total_energy, d, d_)
    # second derivative of energy with respect to d
    J_dam = derivative(F_dam, d, dd)
    # Definition of the optimisation problem with respect to d
    damage_problem = TAOProblem(total_energy, F_dam, J_dam, d, bc_d)

    b = la.create_petsc_vector(V_d.dofmap.index_map, V_d.dofmap.index_map_bs)
    J = fem.petsc.create_matrix(damage_problem.a)

    # Create PETSc TAO
    solver_d_tao = PETSc.TAO().create()
    solver_d_tao.setType("tron")
    solver_d_tao.setObjective(damage_problem.f)
    solver_d_tao.setGradient(damage_problem.F, b)
    solver_d_tao.setHessian(damage_problem.J, J)
    solver_d_tao.setTolerances(grtol=1e-6, gttol=1e-6)
    solver_d_tao.getKSP().setType("preonly")
    solver_d_tao.getKSP().getPC().setType("lu")

    # We set the bound (Note: they are passed as reference and not as values)
    solver_d_tao.setVariableBounds(dlb.vector, dub.vector)

    Nincr = len(load_steps) - 1

    vol = integrate(fem.Constant(domain, 1.0) * dx)
    results = np.zeros((Nincr + 1, 4))
    t = 0
    iter_log = []
    for i, t in enumerate(load_steps[1:]):
        Uimp.value = t
        dirichlet.update()

        incr_msg = "Increment {:3d}".format(i + 1)
        iter_log.append(incr_msg)
        print(incr_msg)

        niter = 0
        for niter in range(Nitermax):
            # Solve displacement
            u_solver.solve(u)
            # Compute new damage
            solver_d_tao.solve(d.vector)

            # check error and update
            L2_error = fem.form(inner(d - dold, d - dold) * dx)
            error_L2 = np.sqrt(fem.assemble_scalar(L2_error)) / vol

            # Update damage
            d.vector.copy(dold.vector)

            iter_msg = "    Iteration {:3d}: ||Res||_2={:5e}".format(niter, error_L2)
            iter_log.append(iter_msg)
            print(iter_msg)
            if error_L2 < tol:
                break
        else:
            warnings.warn("Too many iterations in fixed point algorithm")

        # Update lower bound to account for irreversibility
        d.vector.copy(dlb.vector)

        # compute force (per unit thickness in z direction) and energies
        Force = integrate(ufl.action(F_u, v_reac))

        en_el = integrate(elastic_energy)
        en_f = integrate(fracture_energy)

        results[i + 1, :] = (t, Force, en_el, en_f)

        # compute stress
        sig = interpolate_expr(sigma(u, d), V_sig, name="Stress")

        # Export to Paraview
        save_to_file(problem_type, d, t=t, rewrite=(i == 0))
        save_to_file(problem_type, u, t=t, rewrite=(i == 0))

        clear_output(wait=True)

        with open("iterations_log.txt", "w") as fp:
            fp.write("\n".join(iter_log))

    return results, u, d, sig
