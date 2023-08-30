import math
import numpy as np
import matplotlib.pyplot as plt
import ufl
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, io, cpp, geometry
from dolfinx.cpp.nls.petsc import NewtonSolver
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.solvers import (
    NonlinearMaterialProblem,
)
from dolfinx_materials.material.mfront import MFrontMaterial
from dolfinx_materials.utils import axi_grad, nonsymmetric_tensor_to_vector
from mealor.utils import integrate, evaluate_on_points, save_to_file
from geometry import generate_NT
from load_stepping import LoadSteppingStrategy
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from implicit_gradient import ImplicitGradient
from mealor import mark_facets, DirichletBoundaryCondition


class LocalNonlinearMaterialProblem(NonlinearMaterialProblem):
    def solve(self, solver):
        solver.setF(self.F, self.vector())
        solver.setJ(self.J, self.matrix())
        solver.set_form(self.form)

        it, converged = solver.solve(self.u.vector)
        self.u.x.scatter_forward()

        return converged, it

    def update(self):
        self.quadrature_map.advance()


## Define geometry and mesh (quadrangles)
refinement_level = 1
R = 0.92            # notch radius
height = 10         # half-height of the specimen
Phi = 2.2           # maximum diameter
Phi_0 = 1.23        # minmum diameter
coarse_size = 0.25   # coarse mesh size
fine_size = 0.025    # fine mesh size
fine_size /= 2**(refinement_level)
domain, cell_markers, facet_markers = generate_NT(
    height, Phi, Phi_0, R, coarse_size, fine_size
)

## Regularization length
lc = 0.05


V = fem.VectorFunctionSpace(domain, ("CG", 1))
deg_quad = 1
ds = ufl.Measure(
    "ds",
    domain=domain,
    subdomain_data=facet_markers,
    metadata={"quadrature_degree": deg_quad},
)

Vk = fem.FunctionSpace(domain, ("CG", 1))
dw_nl = fem.Function(Vk, name="NonLocalPlasticVolumeIncrement")
dk_nl = fem.Function(Vk, name="NonLocalEquivalentPlasticStrainIncrement")

# The ImplicitGradient class is used to solve the scalar Helmoltz equation
#
# \[
# - l_{c}^{2}\Delta\,\bar{x} + \bar{x} = x
# \]

lc1 = lc
ell1 = fem.Constant(domain, lc1)
smoother1 = ImplicitGradient(domain, ell1, degree=deg_quad, axisymmetrical=True)

lc2 = lc
ell2 = fem.Constant(domain, lc2)
smoother2 = ImplicitGradient(domain, ell2, degree=deg_quad, axisymmetrical=True)

x = ufl.SpatialCoordinate(domain)

# Load the mechanical behaviour generated by MFront
material = MFrontMaterial(
    "src/libBehaviour.so",
    "ImplicitGradientGTN",
    material_properties={},
)

# mechanical problem
## Boundary conditions
Uimp = fem.Constant(domain, 1.0)
dirichlet = DirichletBoundaryCondition(V)
dirichlet.add_bc_topological(facet_markers, 1, uy=0)
dirichlet.add_bc_topological(facet_markers, 2, ux=0, uy=Uimp)
dirichlet.add_bc_topological(facet_markers, 3, ux=0)


## Test, trial and unknown function
v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)
u = fem.Function(V, name="Displacement")

# Dummy function used to compute force reaction
# fill with u=1 on imposed displacement boundary and use residual
v_reac = fem.Function(V)
fem.set_bc(v_reac.vector, dirichlet.bcs)

# Definition of the deformation gradient
#
# Here, we compute a 3D tensor to integrate the
# behaviour in 3D
def F(u):
    return nonsymmetric_tensor_to_vector(ufl.Identity(3) + axi_grad(x[0], u))


qmap = QuadratureMap(domain, deg_quad, material)
qmap.register_gradient("DeformationGradient", F(u))


PK1 = qmap.fluxes["FirstPiolaKirchhoffStress"]
Res = ufl.dot(PK1, ufl.derivative(F(u), u, v)) * x[0] * qmap.dx
Jac = qmap.derivative(Res, u, du)

tol = 1e-6
problem = LocalNonlinearMaterialProblem(qmap, Res, Jac, u, dirichlet.bcs)

# This passes the non local variables as external state variables to the mechanical behaviour
qmap.register_external_state_variable("NonLocalPlasticVolumeIncrement", dw_nl)
qmap.register_external_state_variable("NonLocalEquivalentPlasticStrainIncrement", dk_nl)

# Create Newton solver and solve

newton = NewtonSolver(MPI.COMM_WORLD)
newton.rtol = tol
newton.report = True
newton.convergence_criterion = "incremental"


## Output file names
prefix = f"nonlocal_mesh_{fine_size}_lc_{lc}"
out_file = prefix + f"/results.xdmf"


# initialize the porosity
f = qmap.internal_state_variables["Porosity"]
fini = 0.004
f.vector.array[:] = fini
qmap.update_initial_state("Porosity")

# Recover broken state variable
broken = qmap.internal_state_variables["Broken"]

## Define load stepping strategy
# (heuristic to adapt time step to target a specific porosity increment `target_df`)
target_df = 1e-3  # target porosity increase
dU_max = 1e-2  # maximum displacement increment
dU_min = 1e-5  # minimum displacement increment
load_stepper = LoadSteppingStrategy(target_df, fini, dU_min, dU_max)

## Load-stepping loop
U_max = 1.5  # final displacement
dU1 = 5e-4  # first displacement increment
U = 0  # initial total displacement
i = 0

dU = dU1
problem_stats = []
jmax = 1
nsub_steps = 0

while U < U_max:
    i += 1
    U += dU
    Uimp.value = U
    dirichlet.update()

    print(f"Increment {i}. Strain {U}")
    
    # Fixed-point algorithm
    #
    # This is *not* an industrial proof implementation
    # as the number of iterations is fixed and does
    # not ensure properly convergence, though it is
    # sufficient for the purpose of this tutorial
    #
    # The displacement problem is solved and
    # the local internal variables is used to
    # update the non local variables \(\bar{\omega}\)
    # and \(\bar{\kappa}\)
    try:
        for j in range(0, jmax):
            if jmax >= 2:
                print(f"iteration {j}, {jmax}")

            # Find a displacement satisfying the equilibrium
            # at fixed non local variables.
            #
            # Note that the internal state variables shall *not*
            # be updated as this resolution may be called several
            # times for the same time step.
            converged, it = problem.solve(newton)

            if not converged:
                break

            # Smooth local state variables increments into nonlocal ones
            dw = qmap.internal_state_variables["PlasticVolumeIncrement"]
            smoother1.smooth(dw, dw_nl)
            dk = qmap.internal_state_variables["EquivalentPlasticStrainIncrement"]
            smoother2.smooth(dk, dk_nl)
            qmap.update_external_state_variables()

    except RuntimeError:
        converged = False
        pass

    # Compute number of broken points
    bp = broken.vector.array[:]
    # Compute maximum porosity for non-broken points
    f_max = max(f.vector.array[np.logical_not(bp)])

    # Trying to handle non-convergence.
    #
    # Probably broken as we do not handle the non local variables
    if not converged:
        nsub_steps += 1
        if nsub_steps >= 10:
            raise RuntimeError
        i -= 1
        U -= dU
        # Update load step
        dU /= 2
        continue

    # After a certain threshold in porosity,
    # increase the number of fixed-point iterations
    if f_max > 0.025:
        jmax = 2

    # Update the state of the material.
    #
    # For instance, the values of the internal states of the
    # variables at the end of the time step is copied on the
    # values at the beginning of the time step
    problem.update()

    num_broken = sum(broken.vector.array[:])
    print("Number of broken points", num_broken)

    # Update load step
    dU = load_stepper.new_step(dU, f_max)

    # Get internal state variables as DG-0 functions for output
    porosity = qmap.project_on("Porosity", ("DG", 0)) 
    p = qmap.project_on("EquivalentPlasticStrain", ("DG", 0))
    
    # Output to files
    rewrite = (i==1)
    save_to_file(out_file, u, t=U, rewrite=rewrite)
    save_to_file(out_file, porosity, t=U, rewrite=rewrite)
    save_to_file(out_file, p, t=U, rewrite=rewrite)

    # Evaluate fields at given points
    uv = evaluate_on_points(u, [Phi_0/2, 0, 0])
    pv = evaluate_on_points(p, [0.,0.,0.])[0]
    DeltaPhi = -2*uv[0]
    
    
    # Nominal stress calculation
    S0 = np.pi * Phi_0**2 / 4
    Force = 2 * np.pi*integrate(ufl.action(Res, v_reac))
    print(
        "Stress [MPa]:",
        Force/S0,
    )

    # Save to file
    problem_stats.append([U / height, f_max, pv, Force / S0, DeltaPhi, num_broken])
    results = np.asarray(problem_stats)
    np.savetxt(
        prefix + f"/results.csv",
        results,
        delimiter=",",
        header="Strain, f_max, p, Sigma, DeltaPhi, num_broken",
    )
