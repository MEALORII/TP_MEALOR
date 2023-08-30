# from implicit_gradient import mealor
# from implicit_gradient import mealor

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
from geometry import generate_NT
from load_stepping import LoadSteppingStrategy
import sys, os
from implicit_gradient import ImplicitGradient

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mealor import mark_facets, DirichletBoundaryCondition

class LocalNonlinearMaterialProblem(NonlinearMaterialProblem):

     def solve2(self, solver):
          solver.setF(self.F, self.vector())
          solver.setJ(self.J, self.matrix())
          solver.set_form(self.form)
          
          it, converged = solver.solve(self.u.vector)
          self.u.x.scatter_forward()
          
          return converged, it

     def update(self):
        self.quadrature_map.advance()
          
## Define geometry and mesh (quadrangles)
R = 4.0            # notch radius
height = 23.0      # half-height of the specimen
Phi = 18.0         # maximum diameter
Phi_0 = 10.0       # minmum diameter
coarse_size = 2.0  # coarse mesh size
fine_size = 0.17   # fine mesh size
domain, cell_markers, facet_markers = generate_NT(height, Phi, Phi_0, R, coarse_size, fine_size)

# with io.XDMFFile(MPI.COMM_WORLD, "nt4.xdmf", "r") as xdmf:
#     domain = xdmf.read_mesh(name="Grid", ghost_mode=cpp.mesh.GhostMode.none)

height = 23.0
width = 9.0

def bottom(x):
     return np.isclose(x[1], 0)

def left(x):
     return np.isclose(x[0], 0)

def top(x):
     return np.isclose(x[1], height)

facet_tag = mark_facets(domain, {1: top, 2: bottom, 3: left})

V = fem.VectorFunctionSpace(domain, ("CG", 2))
deg_quad = 2
ds = ufl.Measure(
    "ds",
    domain=domain,
    subdomain_data=facet_tag,
    metadata={"quadrature_degree": deg_quad},
)

Vk = fem.FunctionSpace(domain, ("CG", 2))
dw_nl = fem.Function(Vk, name="NonLocalPlasticVolumeIncrement")
dk_nl = fem.Function(Vk, name="NonLocalEquivalentPlasticStrainIncrement")

lc = 0.4

lc1 = lc
ell1 = fem.Constant(domain, lc1)
smoother1 = ImplicitGradient(domain, ell1, degree=deg_quad, axisymmetrical = True)

lc2 = lc
ell2 = fem.Constant(domain, lc2)
smoother2 = ImplicitGradient(domain, ell2, degree=deg_quad, axisymmetrical = True)

x = ufl.SpatialCoordinate(domain)

material = MFrontMaterial(
     "src/libBehaviour.so",
     "ImplicitGradientGTN",
     material_properties={
    },
)

# mechanical problem

Uimp = fem.Constant(domain, 0.0)
dirichlet = DirichletBoundaryCondition(V)
dirichlet.add_bc_geometrical(left, ux=0)
dirichlet.add_bc_geometrical(bottom, uy=0)
dirichlet.add_bc_geometrical(top, ux=0, uy=Uimp)

du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
u = fem.Function(V, name="Displacement")

def F(u):
    return nonsymmetric_tensor_to_vector(ufl.Identity(3) + axi_grad(x[0], u))

qmap = QuadratureMap(domain, deg_quad, material)
qmap.register_gradient("DeformationGradient", F(u))


PK1 = qmap.fluxes["FirstPiolaKirchhoffStress"]
Res = ufl.dot(PK1, ufl.derivative(F(u), u, v)) * x[0] * qmap.dx
Jac = qmap.derivative(Res, u, du)

tol = 1e-6
problem = LocalNonlinearMaterialProblem(qmap, Res, Jac, u, dirichlet.bcs)

#
qmap.register_external_state_variable("NonLocalPlasticVolumeIncrement", dw_nl)
qmap.register_external_state_variable("NonLocalEquivalentPlasticStrainIncrement", dk_nl)

# Create Newton solver and solve

newton = NewtonSolver(MPI.COMM_WORLD)
newton.rtol = tol
newton.report = True
newton.convergence_criterion = "incremental"

out_file = "GTN/results.xdmf"
file_results = io.XDMFFile(
    domain.comm,
     out_file,
     "w",
)
file_results.write_mesh(domain)
file_results.close()


f = qmap.internal_state_variables["Porosity"]
fini=0.004
f.vector.array[:] = fini
qmap.update_initial_state("Porosity")
# Recover broken state variable
broken = qmap.internal_state_variables["Broken"]

## Define load stepping strategy 
# (heuristic to adapt time step to target a specific porosity increment `target_df`)
target_df = 5e-4  # target porosity increase
dU_max = 1e-2  # maximum displacement increment
dU_min = 1e-7  # minimum displacement increment
load_stepper = LoadSteppingStrategy(target_df, fini, dU_min, dU_max)

## Load-stepping loop
U_max = 1.5  # final displacement
dU1 = 4e-3  # first displacement increment
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
    try:
         for j in range(0, jmax):
              
              if jmax >= 2:
                   print(f"iteration {j}, {jmax}")
               
              converged, it = problem.solve2(newton)
              
              if not converged:
                   break
              
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
    
    if not converged:
         nsub_steps += 1
         if nsub_steps >= 10:
              raise RuntimeError
         i -= 1
         U -= dU
         # Update load step
         dU /= 2
         continue

    if f_max > 0.06:
         jmax = 2
    if f_max > 0.2:
         jmax = 4
         
    problem.update()
    
    num_broken = sum(broken.vector.array[:])
    print("Number of broken points", num_broken)
    
    # Update load step
    dU = load_stepper.new_step(dU, f_max)
    
    PK1_0 = qmap.project_on("FirstPiolaKirchhoffStress", ("DG", 0))
    
    s = fem.assemble_scalar(fem.form(PK1_0[2] * 2 * np.pi * x[0] * ds(1))) * 1e-3
    Force = domain.comm.allreduce(s, op=MPI.SUM)
    print(
         "Force [kN]:",
         Force,
    )
    
    #Sxx[i + 1] = PK1.vector.array[1]
    
    porosity = qmap.project_on("Porosity", ("DG", 0))  # porosity as DG-0 function
    p = qmap.project_on("EquivalentPlasticStrain", ("DG", 0))
         
    x0 = [5.0, 0, 0.0]
    tree = geometry.BoundingBoxTree(domain, 2)

    cell_candidates = geometry.compute_collisions(tree, x0)  # get candidates
    colliding_cells = geometry.compute_colliding_cells(
        domain, cell_candidates, x0
    )  # get actual

    with io.XDMFFile(domain.comm, out_file, "a") as file_results:
          file_results.write_function(u, i)
          file_results.write_function(qmap.project_on("EquivalentPlasticStrain", ("DG", 0)), i)
          file_results.write_function(qmap.project_on("Porosity", ("DG", 0)), i)

    if MPI.COMM_WORLD.rank == 0:

        DeltaPhi = MPI.COMM_WORLD.gather(-2 * u.eval(x0, colliding_cells)[0], root=0)[0]
        problem_stats.append([i, it, f_max, Force, DeltaPhi])
        
        results = np.asarray(problem_stats)
        np.savetxt(
          f"results-h_{fine_size}.csv",
          results,
          delimiter=" ",
        )

# plt.figure()
# plt.plot(Exx, Sxx, "-o")
# plt.xlabel(r"Strain $\varepsilon_{xx}$")
# plt.ylabel(r"Stress $\sigma_{xx}$")
# plt.savefig(f"{material.name}_stress_strain.pdf")
