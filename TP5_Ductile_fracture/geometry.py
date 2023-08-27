#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Generate geometry and mesh using GMSH for a single-edge notched beam (right half only)

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   26/05/2023
"""
from dolfinx.io import gmshio
from mpi4py import MPI
import gmsh


def generate_NT(L, Phi, Phi_0, R, coarse_size, fine_size):
    """_summary_

    Parameters
    ----------
    L : float
        half-length of the specimen
    Phi : float
        large diameter of the specimen
    Phi_0 : float
        diameter of the central region
    R : float
        notch radius
    coarse_size : float
        coarse mesh size away from crack tip
    fine_size : float
        fine mesh size around crack tip

    Returns
    -------
    _mesh, domain_markers, facet_markers
    """
    try:
        return generate_NT_struct(L, Phi, Phi_0, R, coarse_size, fine_size)
    except AssertionError:
        gmsh.finalize()
        return generate_NT_unstruct(L, Phi, Phi_0, R, coarse_size, fine_size)


def generate_NT_struct(L, Phi, Phi_0, R, coarse_size, fine_size):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)
    # gmsh.option.setNumber("Mesh.Algorithm", 8)
    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    occ = gmsh.model.occ
    W = Phi / 2
    W0 = Phi_0 / 2
    Nx = int((W0 - 4 * fine_size) / fine_size) + 1
    Ny = 1

    if W0 + R < W:
        b = R
    else:
        b = (R**2 - (W0 + R - W) ** 2) ** 0.5
    if mesh_comm.rank == model_rank:
        points = [
            occ.add_point(0, 0, 0),
            occ.add_point(W0, 0, 0),
            occ.add_point(W0 + R, 0, 0),
            occ.add_point(W, b, 0),
            occ.add_point(W, L, 0),
            occ.add_point(0, L, 0),
            occ.add_point(0, Ny * fine_size, 0),
            occ.add_point(W0 - 4 * fine_size, 0, 0),
            occ.add_point(W0 - 4 * fine_size, Ny * fine_size, 0),
        ]
        lines_main = [
            occ.add_line(points[7], points[1]),
            occ.add_circle_arc(points[1], points[2], points[3]),
            occ.add_line(points[3], points[4]),
            occ.add_line(points[4], points[5]),
            occ.add_line(points[5], points[6]),
        ]
        lines_ref = [
            occ.add_line(points[0], points[7]),
            occ.add_line(points[7], points[8]),
            occ.add_line(points[8], points[6]),
            occ.add_line(points[6], points[0]),
        ]
        line_loop_ref = occ.add_curve_loop(lines_ref)
        line_loop_main = occ.add_curve_loop(lines_main + [-lines_ref[2], -lines_ref[1]])

        occ.synchronize()

        gmsh.model.addPhysicalGroup(
            1, [lines_ref[0], lines_main[0]], 1
        )  # bottom support
        gmsh.model.addPhysicalGroup(1, [lines_main[3]], 2)  # top support
        gmsh.model.addPhysicalGroup(1, [lines_main[4], lines_ref[3]], 3)  # left support

        specimen = occ.add_plane_surface([line_loop_main])
        specimen_ref = occ.add_plane_surface([line_loop_ref])
        occ.synchronize()
        gmsh.model.addPhysicalGroup(gdim, [specimen, specimen_ref], 1)

        field_tag = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field_tag, "VIn", fine_size)
        gmsh.model.mesh.field.setNumber(field_tag, "VOut", coarse_size)
        gmsh.model.mesh.field.setNumber(field_tag, "XMin", 0)
        gmsh.model.mesh.field.setNumber(field_tag, "XMax", W)
        gmsh.model.mesh.field.setNumber(field_tag, "YMin", 0)
        gmsh.model.mesh.field.setNumber(field_tag, "YMax", W - W0)
        gmsh.model.mesh.field.setNumber(field_tag, "Thickness", W - W0)

        gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

        gmsh.model.mesh.set_transfinite_surface(specimen_ref)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.set_recombine(2, specimen)
        gmsh.model.mesh.set_recombine(2, specimen_ref)

        gmsh.model.mesh.generate(2)

        mesh, ct, ft = gmshio.model_to_mesh(
            gmsh.model, mesh_comm, model_rank, gdim=gdim
        )

        gmsh.finalize()

    return mesh, ct, ft


def generate_NT_unstruct(L, Phi, Phi_0, R, coarse_size, fine_size):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    occ = gmsh.model.occ
    W = Phi / 2
    W0 = Phi_0 / 2
    if W0 + R < W:
        b = R
    else:
        b = (R**2 - (W0 + R - W) ** 2) ** 0.5
    if mesh_comm.rank == model_rank:
        points = [
            occ.add_point(0, 0, 0),
            occ.add_point(W0, 0, 0),
            occ.add_point(W0 + R, 0, 0),
            occ.add_point(W, b, 0),
            occ.add_point(W, L, 0),
            occ.add_point(0, L, 0),
        ]
        lines = [
            occ.add_line(points[0], points[1]),
            occ.add_circle_arc(points[1], points[2], points[3]),
            occ.add_line(points[3], points[4]),
            occ.add_line(points[4], points[5]),
            occ.add_line(points[5], points[0]),
        ]
        line_loop = occ.add_curve_loop(lines)

        occ.synchronize()

        gmsh.model.addPhysicalGroup(1, [lines[0]], 1)  # bottom support
        gmsh.model.addPhysicalGroup(1, [lines[3]], 2)  # top support
        gmsh.model.addPhysicalGroup(1, [lines[4]], 3)  # left support

        specimen = occ.add_plane_surface([line_loop])
        occ.synchronize()
        gmsh.model.addPhysicalGroup(gdim, [specimen], 1)

        field_tag = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field_tag, "VIn", fine_size)
        gmsh.model.mesh.field.setNumber(field_tag, "VOut", coarse_size)
        gmsh.model.mesh.field.setNumber(field_tag, "XMin", 0)
        gmsh.model.mesh.field.setNumber(field_tag, "XMax", W)
        gmsh.model.mesh.field.setNumber(field_tag, "YMin", 0)
        gmsh.model.mesh.field.setNumber(field_tag, "YMax", W - W0)
        gmsh.model.mesh.field.setNumber(field_tag, "Thickness", W - W0)

        gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

        gmsh.model.mesh.set_recombine(2, specimen)

        gmsh.model.mesh.generate(2)
        gmsh.write("NT4.msh")

        mesh, ct, ft = gmshio.model_to_mesh(
            gmsh.model, mesh_comm, model_rank, gdim=gdim
        )

        gmsh.finalize()

    return mesh, ct, ft
