from dolfinx import io, geometry
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI
import numpy as np
import gmsh
from mealor.utils import mark_facets


def _generate_edge_crack_mesh(
    L, W, a, theta, ex, ey, W_clamp, h_clamp, fine_size, coarse_size, double=False
):
    gmsh.initialize()

    ax = a

    occ = gmsh.model.occ
    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        clamp_b = occ.addRectangle(-W_clamp / 2, -L / 2, 0, W_clamp, h_clamp)
        clamp_t = occ.addRectangle(-W_clamp / 2, L / 2 - h_clamp, 0, W_clamp, h_clamp)
        central = occ.addRectangle(-W / 2, -L / 2 + h_clamp, 0, W, L - 2 * h_clamp)
        crack = occ.addRectangle(-W / 2 - ex / 2 - ax, -ey / 2, 0, 2 * ax, ey)
        notch = occ.add_disk(-W / 2 + ax - ex / 2, 0, 0, ex / 2, ey / 2)

        occ.synchronize()

        out, _ = occ.fuse([(gdim, crack)], [(gdim, notch)])
        crack_notched = out[0][1]

        if double:
            theta = 0.0
            out = occ.copy([(gdim, crack_notched)])
            crack_2 = out[0][1]
            occ.mirror(out, 1, 0, 0, 0)

            out, _ = occ.cut(
                [(gdim, central)], [(gdim, crack_notched), (gdim, crack_2)]
            )
            central_cracked = out[0][1]
        else:
            occ.rotate([(gdim, crack)], -W / 2 + ax, 0.0, 0.0, 0.0, 0.0, 1.0, theta)

            out, _ = occ.cut([(gdim, central)], [(gdim, crack)])
            central_cracked = out[0][1]

        occ.synchronize()

        occ.fuse([(gdim, clamp_b)], [(gdim, clamp_t), (gdim, central_cracked)])

        occ.synchronize()

        gmsh.write("SENT.geo_unrolled")

        volumes = gmsh.model.getEntities(gdim)
        assert len(volumes) == 1
        gmsh.model.addPhysicalGroup(gdim, [volumes[0][1]], 1)

        gmsh.model.setPhysicalName(gdim, 1, "Plate")
        field_tag = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field_tag, "VIn", fine_size)
        gmsh.model.mesh.field.setNumber(field_tag, "VOut", coarse_size)
        gmsh.model.mesh.field.setNumber(field_tag, "XMin", -W / 2)
        gmsh.model.mesh.field.setNumber(field_tag, "XMax", W / 2)
        gmsh.model.mesh.field.setNumber(field_tag, "YMin", -W / 8)
        gmsh.model.mesh.field.setNumber(field_tag, "YMax", W / 8)
        gmsh.model.mesh.field.setNumber(field_tag, "Thickness", W / 4)

        gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

        gmsh.model.mesh.generate(gdim)

        gmsh.write("SENT.msh")

        mesh, ct, ft = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
        ft.name = "Facet markers"

        gmsh.finalize()
    return mesh, ct, ft


def _generate_CT_mesh(W, a, e, theta, coarse_size, fine_size, a_kink):
    L = 1.25 * W
    H = 1.2 * W
    a = 3.0
    x = 0.25 * W
    y = 0.275 * W
    R = 0.125 * W

    gmsh.initialize()
    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    occ = gmsh.model.occ
    if mesh_comm.rank == model_rank:
        # gmsh.moccodel.occ.

        points = [
            occ.add_point(0, -H / 2, 0),
            occ.add_point(L, -H / 2, 0),
            occ.add_point(L, H / 2, 0),
            occ.add_point(0, H / 2, 0),
        ]
        lines = [
            occ.add_line(points[0], points[1]),
            occ.add_line(points[1], points[2]),
            occ.add_line(points[2], points[3]),
            occ.add_line(points[3], points[0]),
        ]
        line_loop = occ.add_curve_loop(lines)

        R2 = R / np.sqrt(2)
        top_circle_points = [
            occ.add_point(x, y, 0),
            occ.add_point(x - R2, y - R2, 0),
            occ.add_point(x + R2, y - R2, 0),
            occ.add_point(x + R2, y + R2, 0),
            occ.add_point(x - R2, y + R2, 0),
        ]
        bot_circle_points = [
            occ.add_point(x, -y, 0),
            occ.add_point(x - R2, -y - R2, 0),
            occ.add_point(x + R2, -y - R2, 0),
            occ.add_point(x + R2, -y + R2, 0),
            occ.add_point(x - R2, -y + R2, 0),
        ]
        top_circle = [
            occ.add_circle_arc(
                top_circle_points[1], top_circle_points[0], top_circle_points[2]
            ),
            occ.add_circle_arc(
                top_circle_points[2], top_circle_points[0], top_circle_points[3]
            ),
            occ.add_circle_arc(
                top_circle_points[3], top_circle_points[0], top_circle_points[4]
            ),
            occ.add_circle_arc(
                top_circle_points[4], top_circle_points[0], top_circle_points[1]
            ),
        ]
        bot_circle = [
            occ.add_circle_arc(
                bot_circle_points[1], bot_circle_points[0], bot_circle_points[2]
            ),
            occ.add_circle_arc(
                bot_circle_points[2], bot_circle_points[0], bot_circle_points[3]
            ),
            occ.add_circle_arc(
                bot_circle_points[3], bot_circle_points[0], bot_circle_points[4]
            ),
            occ.add_circle_arc(
                bot_circle_points[4], bot_circle_points[0], bot_circle_points[1]
            ),
        ]
        top_circle_loop = occ.add_curve_loop(top_circle)
        bot_circle_loop = occ.add_curve_loop(bot_circle)
        specimen = occ.add_plane_surface([line_loop, top_circle_loop, bot_circle_loop])

        top_pin_lines = [
            occ.add_line(top_circle_points[0], top_circle_points[3]),
            top_circle[2],
            occ.add_line(top_circle_points[4], top_circle_points[0]),
        ]
        top_pin_loop = occ.add_curve_loop(top_pin_lines)
        top_pin = occ.add_plane_surface([top_pin_loop])

        bot_pin_lines = [
            occ.add_line(bot_circle_points[0], bot_circle_points[1]),
            bot_circle[0],
            occ.add_line(bot_circle_points[2], bot_circle_points[0]),
        ]
        bot_pin_loop = occ.add_curve_loop(bot_pin_lines)
        bot_pin = occ.add_plane_surface([bot_pin_loop])

        occ.synchronize()

        a_tot = a + x
        lx_tip = a_kink * a_tot
        a_s = a_tot - lx_tip
        y_tip = -lx_tip * np.sin(theta)

        crack = occ.addRectangle(0, -e / 2, 0, a_s, e)
        hinge = occ.addDisk(a_s, 0, 0, e / 2, e / 2)
        out, _ = occ.fuse([(gdim, crack)], [(gdim, hinge)], removeTool=False)
        straight_crack = out[0][1]

        crack_45 = occ.addRectangle(a_s, -e / 2, 0, lx_tip - e / 2, e)
        notch = occ.addDisk(a_tot - e / 2, 0, 0, e / 2, e / 2)
        out, _ = occ.fuse([(gdim, crack_45)], [(gdim, notch), (gdim, hinge)])
        tip = out[0][1]

        occ.rotate([(gdim, tip)], a_s, 0, 0, 0, 0, 1, -theta)
        out, _ = occ.cut([(gdim, specimen)], [(gdim, straight_crack), (gdim, tip)])
        specimen = out[0][1]

        occ.synchronize()
        gmsh.model.addPhysicalGroup(gdim, [specimen], 1)
        gmsh.model.addPhysicalGroup(gdim, [bot_pin, top_pin], 2)
        gmsh.model.addPhysicalGroup(1, bot_pin_lines, 1)
        gmsh.model.addPhysicalGroup(1, top_pin_lines, 2)

        field_tag = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field_tag, "VIn", fine_size)
        gmsh.model.mesh.field.setNumber(field_tag, "VOut", coarse_size)
        gmsh.model.mesh.field.setNumber(field_tag, "XMin", a_tot - 2 * e)
        gmsh.model.mesh.field.setNumber(field_tag, "XMax", L)
        gmsh.model.mesh.field.setNumber(field_tag, "YMin", y_tip - 4 * e)
        gmsh.model.mesh.field.setNumber(field_tag, "YMax", y_tip + 4 * e)
        gmsh.model.mesh.field.setNumber(field_tag, "Thickness", e)

        gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

        gmsh.model.mesh.generate(2)

        gmsh.write("CT.geo_unrolled")
        mesh, ct, ft = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
        ct.name = "Cell markers"
        ft.name = "Facet markers"

        gmsh.finalize()

    return mesh, ct, ft


def generate_edge_notch_tension(
    problem, L, W, a, theta, fine_size, coarse_size, e=0.25
):
    W_clamp = 2 * W
    h_clamp = W / 2
    if problem == "SENT":
        domain, subdomains, facets = _generate_edge_crack_mesh(
            L, W, a, theta, e, e, W_clamp, h_clamp, fine_size, coarse_size, False
        )
    elif problem == "DENT":
        domain, subdomains, facets = _generate_edge_crack_mesh(
            L, W, a, 0.0, e, e, W_clamp, h_clamp, fine_size, coarse_size, True
        )

    # Define boundaries and boundary integration measure
    def bottom(x):
        return np.isclose(x[1], -L / 2, atol=0)

    def top(x):
        return np.isclose(x[1], L / 2, atol=0)

    boundaries = {1: bottom, 2: top}
    facet_tag = mark_facets(domain, boundaries)
    return domain, subdomains, facet_tag


def generate_compact_tension(
    problem, W, a, e, theta, coarse_size, fine_size, a_kink=0.17
):
    if problem == "CT":
        return _generate_CT_mesh(W, a, e, 0, coarse_size, fine_size, a_kink)
    elif problem == "CT-kink":
        return _generate_CT_mesh(W, a, e, theta, coarse_size, fine_size, a_kink)
