from dolfinx import fem, plot
import numpy as np
import pyvista as pv
import warnings

warnings.filterwarnings("ignore")

pv.global_theme.enable_camera_orientation_widget = True
pv.set_plot_theme("paraview")
pv.start_xvfb(wait=0.1)
pv.set_jupyter_backend("panel")  # "panel" for interaction
pv.global_theme.font.color = "black"


def get_grid(V):
    topology, cell_types, geometry = plot.create_vtk_mesh(V)
    dim = V.mesh.topology.dim
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    return grid, dim


def plot_mesh(mesh):
    """Plot DOLFIN mesh."""

    V = fem.FunctionSpace(mesh, ("CG", 1))
    grid, dim = get_grid(V)

    # Create plotter and pyvista grid
    p = pv.Plotter()

    p.add_mesh(grid, show_edges=True)
    p.view_xy()
    p.show_axes()
    p.screenshot("mesh.png", transparent_background=True)
    if not pv.OFF_SCREEN:
        p.show()


def plot_def(u, scale=1.0, **kwargs):
    V = u.function_space
    topology, cell_types, geometry = plot.create_vtk_mesh(V)
    dim = V.mesh.topology.dim
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    # Create plotter and pyvista grid
    p = pv.Plotter()

    # Attach vector values to grid and warp grid by vector
    u_3D = np.zeros((geometry.shape[0], 3))
    u_3D[:, :dim] = u.x.array.reshape((geometry.shape[0], dim))
    grid["u"] = u_3D
    p.add_mesh(grid, style="surface", color="gray", opacity=0.5)
    warped = grid.warp_by_vector("u", factor=scale)
    p.add_mesh(warped, show_edges=False, cmap="viridis")
    p.show_axes()
    if dim == 2:
        p.view_xy()
    p.screenshot("deformation.png", transparent_background=True)
    if not pv.OFF_SCREEN:
        p.show()


def plot_stress(stress, mesh, clim=None):
    """Plot scalar stress."""

    V_sig = fem.FunctionSpace(mesh, ("DG", 0))
    V = fem.FunctionSpace(mesh, ("CG", 1))
    sig = interpolate_expr(stress, V_sig)

    grid, dim = get_grid(V)

    # Create plotter and pyvista grid
    p = pv.Plotter()

    # Attach vector values to grid and warp grid by vector
    grid.cell_data["Stress"] = sig.vector.array
    grid.set_active_scalars("Stress")

    print(
        f"Stress: (min) {sig.vector.array.min():.4f} -- {sig.vector.array.max():.4f} (max)"
    )

    p.add_mesh(
        grid,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Stress", "interactive": True},
        clim=clim,
        cmap="bwr",
    )
    if dim == 2:
        p.view_xy()
    p.show_axes()
    p.screenshot("stresses.png", transparent_background=True)
    if not pv.OFF_SCREEN:
        p.show()
