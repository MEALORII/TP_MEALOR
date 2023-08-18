#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   26/07/2023
"""
import numpy as np
from dolfinx import mesh, fem, io
import dolfinx.geometry
import os


def integrate(expr):
    return fem.assemble_scalar(fem.form(expr))


def mark_facets(domain, surfaces_dict):
    """Mark facets of the domain according to geometrical location

    Parameters
    ----------
    domain : Mesh
        Dolfinx mesh object
    surfaces_dict : dict
        A dictionnary mapping integer tags with geometrical location function {tag: locator(x)}

    Returns
    -------
    facet_tag array
    """
    fdim = domain.topology.dim - 1
    marked_values = []
    marked_facets = []
    # Concatenate and sort the arrays based on facet indices
    for tag, location in surfaces_dict.items():
        facets = mesh.locate_entities_boundary(domain, fdim, location)
        marked_facets.append(facets)
        marked_values.append(np.full_like(facets, tag))
    marked_facets = np.hstack(marked_facets)
    marked_values = np.hstack(marked_values)
    sorted_facets = np.argsort(marked_facets)
    facet_tag = mesh.meshtags(
        domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
    )
    return facet_tag


def evaluate_on_points(field, points):
    """This function returns the values of a field on a set of points

    Parameters
    ==========
    field: The FEniCS function from which we want points: a n x 3 np.array
    with the coordinates of the points where to evaluate the function

    It returns:
    - points_on_proc: the local slice of the point array
    - values_on_proc: the local slice of the values
    """
    points = np.asarray(points)
    mesh = field.function_space.mesh
    bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    # for each point, compute a colliding cells and append to the lists
    points_on_proc = []
    cells = []
    cell_candidates = dolfinx.geometry.compute_collisions(
        bb_tree, points
    )  # get candidates
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        mesh, cell_candidates, points
    )  # get actual
    if len(points.shape) == 1:
        return field.eval(points, colliding_cells)
    else:
        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                cc = colliding_cells.links(i)[0]
                points_on_proc.append(point)
                cells.append(cc)
        # convert to numpy array
        points_on_proc = np.array(points_on_proc)
        cells = np.array(cells)
        values_on_proc = field.eval(points_on_proc, cells)
        return values_on_proc.ravel(), points_on_proc


def interpolate_expr(expr, V, name=""):
    if isinstance(expr, fem.Function):
        return expr
    else:
        f = fem.Function(V, name=name)
        f_expr = fem.Expression(expr, V.element.interpolation_points())
        f.interpolate(f_expr)
    return f


def plot_over_line(expr, line, mesh, N=500, interp=("CG", 1)):
    points = np.zeros((N + 1, 3))
    points[:, 0] = np.linspace(line[0][0], line[1][0], N + 1)
    points[:, 1] = np.linspace(line[0][1], line[1][1], N + 1)
    return get_over_line(expr, points, mesh, interp)


def get_over_line(expr, points, mesh, interp):
    V = fem.FunctionSpace(mesh, interp)
    fun = interpolate_expr(expr, V)
    return evaluate_on_points(fun, points)


def save_to_file(filename, u, t=0, rewrite=True):
    if not filename.endswith(".xdmf"):
        filename += f"_{u.name}.xdmf"
    else:
        filename = filename.replace(".xdmf", f"_{u.name}.xdmf")

    domain = u.function_space.mesh

    if not os.path.isfile(filename) or rewrite:
        with io.XDMFFile(domain.comm, filename, "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(u, t)
    else:
        with io.XDMFFile(domain.comm, filename, "a") as xdmf:
            xdmf.write_function(u, t)
