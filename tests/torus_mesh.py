import numpy as np
import pyvista as pv

from min_energy_points.torus import SpiralTorus
from min_energy_points.local_voronoi import LocalSurfaceVoronoi

from scipy.spatial import KDTree

N = 48502
# N = 3_000

torus = SpiralTorus(N)
plotter = pv.Plotter(off_screen=False)
plotter.add_mesh(
    pv.PolyData(torus.points),
    style="points",
    render_points_as_spheres=True,
    point_size=20,
)
# plotter.add_mesh(
#     pv.PolyData(torus.points[np.array([0, 15022])]),
#     style="points",
#     render_points_as_spheres=True,
#     point_size=20,
#     color="red",
# )
# plotter.set_focus([4, 0, 0])
plotter.show()

tree = KDTree(torus.points)
print(tree.query(torus.points, k=5)[0])
print(tree.query(torus.points, k=5)[1])

vor = LocalSurfaceVoronoi(torus.points, torus.normals, torus.implicit_surf)
plotter = pv.Plotter(off_screen=False)
surf = pv.PolyData(torus.points, [(3, *f) for f in vor.triangles])
plotter.add_mesh(
    surf,
    show_vertices=True,
    show_edges=True,
    cmap=["#AAAAAA", "#005500", "#774444"],
    show_scalar_bar=False,
)
plotter.show()
