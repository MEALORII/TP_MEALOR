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


def generate_SENB(
    L, H, notch, a, e, s, coarse_size, fine_size, J_contour=True, rJ=None
):
    """_summary_

    Parameters
    ----------
    L : float
        span of the half-beam
    H : float
        height of the half-beam
    notch : float
        depth of the notch
    a : float
        crack extent from the notch tip
    e : float
        half-width of the notch
    s : float
        width of loading and support conditions
    coarse_size : float
        coarse mesh size away from crack tip
    fine_size : float
        fine mesh size around crack tip
    J_contour : bool, optional
        use additional internal contours for J-integral, by default True
    rJ : float, optional
        width of J-integral contour, by default 5*e

    Returns
    -------
    _type_
        _description_
    """
    if rJ is None:
        rJ = 5 * e

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    occ = gmsh.model.occ

    if mesh_comm.rank == model_rank:
        points = [
            occ.add_point(e, 0, 0),
            occ.add_point(L - s, 0, 0, meshSize=fine_size),
            occ.add_point(L, 0, 0, meshSize=fine_size),
            occ.add_point(L, H, 0),
            occ.add_point(s, H, 0, meshSize=fine_size),
            occ.add_point(0, H, 0, meshSize=fine_size),
            occ.add_point(0, (H + a + notch) / 2, 0),
            occ.add_point(0, a + notch, 0),
            occ.add_point(0, notch, 0),
            occ.add_point(e, notch - e, 0),
        ]
        points_J = [
            occ.add_point(rJ, notch, 0),
            occ.add_point(rJ, (H + a + notch) / 2, 0),
        ]
        lines = [
            occ.add_line(points[i], points[i + 1]) for i in range(len(points) - 1)
        ] + [occ.add_line(points[-1], points[0])]

        lines_J = [
            occ.add_line(points[-2], points_J[0]),
            occ.add_line(points_J[0], points_J[1]),
            occ.add_line(points_J[1], points[6]),
        ]
        J_loop = occ.add_curve_loop(lines_J + [lines[6], lines[7]])
        line_loop = occ.add_curve_loop(lines)

        occ.synchronize()

        gmsh.model.addPhysicalGroup(1, [lines[1]], 1)  # bottom right support
        gmsh.model.addPhysicalGroup(1, [lines[4]], 2)  # loading support
        gmsh.model.addPhysicalGroup(1, lines[5:7], 3)  # ahead of crack
        gmsh.model.addPhysicalGroup(1, [lines[7]], 4)  # crack
        gmsh.model.mesh.set_transfinite_curve(lines[1], 5)
        gmsh.model.mesh.set_transfinite_curve(lines[4], 5)

        if J_contour:
            beam = occ.add_plane_surface([line_loop, J_loop])
            beam_J_loop = occ.add_plane_surface([J_loop])
            occ.synchronize()

            gmsh.model.addPhysicalGroup(gdim, [beam, beam_J_loop], 1)
            gmsh.model.addPhysicalGroup(1, [lines_J[0]], 5)  # boundary for J-integral
            gmsh.model.addPhysicalGroup(1, [lines_J[1]], 6)  # boundary for J-integral
            gmsh.model.addPhysicalGroup(1, [lines_J[2]], 7)  # boundary for J-integral
        else:
            beam = occ.add_plane_surface([line_loop])
            occ.synchronize()
            gmsh.model.addPhysicalGroup(gdim, [beam], 1)

        field_tag = gmsh.model.mesh.field.add("Ball")
        gmsh.model.mesh.field.setNumber(field_tag, "VIn", fine_size)
        gmsh.model.mesh.field.setNumber(field_tag, "VOut", coarse_size)
        gmsh.model.mesh.field.setNumber(field_tag, "XCenter", 0)
        gmsh.model.mesh.field.setNumber(field_tag, "YCenter", a + notch)
        gmsh.model.mesh.field.setNumber(field_tag, "ZCenter", 0)
        gmsh.model.mesh.field.setNumber(field_tag, "Radius", 2 * e)
        gmsh.model.mesh.field.setNumber(field_tag, "Thickness", 2 * e)

        gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

        gmsh.model.mesh.generate(2)
        gmsh.write("SENB.geo_unrolled")

        mesh, ct, ft = gmshio.model_to_mesh(
            gmsh.model, mesh_comm, model_rank, gdim=gdim
        )

        gmsh.finalize()

    return mesh, ct, ft
