import dolfinx
from dolfinx import io
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI
import numpy as np
import gmsh
from mealor.utils import mark_facets


def generate_notched_plate(L, W, R, notch_spacing, aspect_ratio, refinement_level):
    gmsh.initialize()

    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, W, tag=1)
        bot_hole = gmsh.model.occ.addDisk(
            L / 2 - notch_spacing,
            0,
            0,
            R,
            R / aspect_ratio,
            zAxis=[0, 0, 1],
            xAxis=[0.0, 1.0, 0.0],
        )
        top_hole = gmsh.model.occ.addDisk(
            L / 2 + notch_spacing,
            W,
            0,
            R,
            R / aspect_ratio,
            zAxis=[0, 0, 1],
            xAxis=[0.0, 1.0, 0.0],
        )
        gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, bot_hole), (gdim, top_hole)])
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(gdim)
        assert len(volumes) == 1
        gmsh.model.addPhysicalGroup(gdim, [volumes[0][1]], 1)
        gmsh.model.setPhysicalName(gdim, 1, "Plate")

        a = ((1 + 1 / aspect_ratio) * R) / 2 + notch_spacing
        coarse_size = W / 20
        fine_size = coarse_size * 2 ** (-refinement_level)
        # Create a new scalar field
        field_tag = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field_tag, "VIn", fine_size)
        gmsh.model.mesh.field.setNumber(field_tag, "VOut", coarse_size)
        gmsh.model.mesh.field.setNumber(field_tag, "XMin", L / 2 - a)
        gmsh.model.mesh.field.setNumber(field_tag, "XMax", L / 2 + a)
        gmsh.model.mesh.field.setNumber(field_tag, "YMin", 0)
        gmsh.model.mesh.field.setNumber(field_tag, "YMax", W)

        gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

        gmsh.model.mesh.generate(gdim)

        mesh, _, ft = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
        ft.name = "Facet markers"

        gmsh.finalize()
    return mesh, ft


def generate_bar(L, W, refinement_level):
    gmsh.initialize()
    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        gmsh.model.occ.addRectangle(0, 0, 0, L, W, tag=1)
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(gdim)
        assert len(volumes) == 1
        gmsh.model.addPhysicalGroup(gdim, [volumes[0][1]], 1)
        gmsh.model.setPhysicalName(gdim, 1, "Plate")

        size = 0.8 * L * 2 ** (-refinement_level)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)

        gmsh.model.mesh.generate(gdim)

        mesh, ct, ft = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
        ft.name = "Facet markers"

        gmsh.finalize()
    return mesh, ft


def generate_shear(L, W, refinement_level):
    a = L / 2

    gmsh.initialize()
    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, W, tag=1)
        crack = gmsh.model.occ.addDisk(
            0,
            L / 2,
            0,
            a,
            a / 100,
        )
        gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, crack)])
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(gdim)
        assert len(volumes) == 1
        gmsh.model.addPhysicalGroup(gdim, [volumes[0][1]], 1)
        gmsh.model.setPhysicalName(gdim, 1, "Plate")

        coarse_size = L / 50
        fine_size = coarse_size * 2 ** (-refinement_level)
        # Create a new scalar field
        field_tag = gmsh.model.mesh.field.add("Ball")
        R = 3 * a / 4
        gmsh.model.mesh.field.setNumber(field_tag, "VIn", fine_size)
        gmsh.model.mesh.field.setNumber(field_tag, "VOut", coarse_size * 2)
        gmsh.model.mesh.field.setNumber(field_tag, "XCenter", a + 3 * R / 4)
        gmsh.model.mesh.field.setNumber(field_tag, "YCenter", L / 2)
        gmsh.model.mesh.field.setNumber(field_tag, "ZCenter", 0)
        gmsh.model.mesh.field.setNumber(field_tag, "Radius", R)

        gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

        gmsh.model.mesh.generate(gdim)

        mesh, _, ft = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
        ft.name = "Facet markers"

        gmsh.finalize()
    return mesh, ft


def setup_geometry(
    problem, L, H, refinement_level=0, notch_spacing=0, notch_radius=0.2, aspect_ratio=1
):
    if problem == "bar":
        if refinement_level == 0:
            domain = dolfinx.mesh.create_rectangle(
                MPI.COMM_WORLD,
                [np.array([0, 0]), np.array([L, H])],
                [1, 1],
            )
        else:
            domain, facets = generate_bar(L, H, refinement_level)

    elif problem == "notched":
        R = notch_radius
        assert (
            2 * R < H
        ), f"Plate height H = {H} is not sufficient for notches of radius R={R}."
        domain, facets = generate_notched_plate(
            L, H, R, notch_spacing, aspect_ratio, refinement_level
        )
    elif problem == "shear":
        domain, facets = generate_shear(L, H, refinement_level)
    else:
        raise ValueError(
            "Unsupport problem type. Supported \
                         problems are 'bar', 'notched', or 'shear'."
        )

    # Define boundaries and boundary integration measure
    def left(x):
        return np.isclose(x[0], 0)

    def right(x):
        return np.isclose(x[0], L)

    def bottom(x):
        return np.isclose(x[1], 0)

    def top(x):
        return np.isclose(x[1], H)

    boundaries = {1: left, 2: bottom, 3: right, 4: top}
    facet_tag = mark_facets(domain, boundaries)
    return domain, facet_tag
