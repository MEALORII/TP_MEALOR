from ufl import (
    grad,
    as_vector,
    inner,
    sqrt,
    dot,
    SpatialCoordinate,
    max_value,
    min_value,
    Measure,
)
from dolfinx import fem
from mealor.utils import integrate, interpolate_expr, save_to_file


def compute_J_integral_(Eshelby, measures):
    Eshelby = Eshelby("+")  # restrict the tensor on a single side
    # Unit vectors
    ex = as_vector([1, 0])
    ey = as_vector([0, 1])
    # Surface integration measure
    dS = measures[2]

    ###
    ### COMPLETE HERE
    ###
    return 0


def compute_G_theta(Eshelby, domain, ell, r_int, r_ext):
    ey = as_vector([0, 1])
    r_ext = fem.Constant(domain, r_ext)
    r_int = fem.Constant(domain, r_int)
    x = SpatialCoordinate(domain)
    V = fem.VectorFunctionSpace(domain, ("CG", 2))

    # Theta field as a function of distance to crack r
    yc = fem.Constant(domain, ell)
    r = sqrt((x[0]) ** 2 + (x[1] - yc) ** 2)
    theta = interpolate_expr(
        max_value(0, min_value((r_ext - r) / (r_ext - r_int), 1)) * ey,
        V,
        name="Theta",
    )
    save_to_file("linear_elasticity", theta)
    dx = Measure("dx", domain=domain)
    
    ### COMPLETE BELOW
    return 0
