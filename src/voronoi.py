import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)

def voronoi_edges(points):
    """
    return voronoi edges
    """
    # print("Points: ", points)
    points = np.array(points)
    vor = Voronoi(points)
    # voronoi_plot_2d(vor)
    # plt.savefig("voro_1.png")
    # plt.show()
    # plot
    regions, vertices = voronoi_finite_polygons_2d(vor)
    # colorize
    voronoi_lines = []
    for region in regions:
        polygon = vertices[region]
        voronoi_lines.append(np.array(np.ceil(polygon), dtype='int'))
        # plt.fill(*zip(*polygon), alpha=0.4)
    # plt.plot(points[:, 0], points[:, 1], 'ko')
    # plt.axis('equal')
    # plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    # plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
    # plt.savefig('voro_2.png')
    # plt.show()
    return np.array(voronoi_lines).tolist()