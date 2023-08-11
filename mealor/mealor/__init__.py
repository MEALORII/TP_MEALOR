from dolfinx import log
import logging

logger = logging.getLogger()
assert len(logger.handlers) == 1
handler = logger.handlers[0]
handler.setLevel(logging.ERROR)

from .boundary_conditions import DirichletBoundaryCondition
from .implicit_gradient import ImplicitGradient
from .utils import mark_facets
from .gmshio import model_to_mesh
