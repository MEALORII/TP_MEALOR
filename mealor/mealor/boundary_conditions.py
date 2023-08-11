#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Convenience class for handling boundary conditions

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   25/07/2023
"""
from dolfinx import fem
import numpy as np


class DirichletBoundaryCondition:
    def __init__(self, V):
        self.bcs = []
        self._V = V
        self._dim = self._V.num_sub_spaces
        self._subspaces = [V.sub(i).collapse()[0] for i in range(self._dim)]
        self._interpolation_mapping = []

    def add_bc_geometrical(self, location, ux=None, uy=None, uz=None):
        u_comp = (ux, uy, uz)
        for i, u in enumerate(u_comp):
            if u is not None:
                dofs = fem.locate_dofs_geometrical(
                    (self._V.sub(i), self._subspaces[i]), location
                )
                u_f = fem.Function(self._subspaces[i])
                if isinstance(u, (int, float, np.ndarray)):
                    u_f.vector.array[:] = u
                else:
                    u_expr = fem.Expression(
                        u, self._subspaces[i].element.interpolation_points()
                    )
                    u_f.interpolate(u_expr)
                    self._interpolation_mapping.append((u_f, u_expr))
                self.bcs.append(fem.dirichletbc(u_f, dofs, self._V.sub(i)))

    def add_bc_topological(self, facets, location, ux=None, uy=None, uz=None):
        u_comp = (ux, uy, uz)
        for i, u in enumerate(u_comp):
            if u is not None:
                dofs = fem.locate_dofs_topological(
                    (self._V.sub(i), self._subspaces[i]),
                    facets.dim,
                    facets.indices[facets.values == location],
                )
                u_f = fem.Function(self._subspaces[i])
                if isinstance(u, (int, float, np.ndarray)):
                    u_f.vector.array[:] = u
                else:
                    u_expr = fem.Expression(
                        u, self._subspaces[i].element.interpolation_points()
                    )
                    u_f.interpolate(u_expr)
                    self._interpolation_mapping.append((u_f, u_expr))
                self.bcs.append(fem.dirichletbc(u_f, dofs, self._V.sub(i)))

    def update(self):
        for u_f, u_expr in self._interpolation_mapping:
            u_f.interpolate(u_expr)
