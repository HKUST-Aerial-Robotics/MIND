import numpy as np
from common.geometry import get_point_line_distance


def gen_dist_field(ego_pos, polyline, discrete_size, resolution):

    field_size = ((discrete_size[0] - 1) * resolution, (discrete_size[1] - 1) * resolution)
    field_offset = np.array([ego_pos[0] - 0.5 * field_size[0], ego_pos[1] - 0.5 * field_size[1]])

    x = np.linspace(0.0, field_size[0], discrete_size[0]) + field_offset[0]
    y = np.linspace(0.0, field_size[1], discrete_size[1]) + field_offset[1]

    xx, yy = np.meshgrid(x, y)
    centroids = np.vstack([xx.ravel(), yy.ravel()]).T

    distance_field = np.full(xx.shape[0] * xx.shape[1], np.inf)

    for j in range(len(polyline) - 1):
        dists = get_point_line_distance(centroids, polyline[j], polyline[j + 1])
        distance_field = np.minimum(distance_field, dists)

    return field_offset, xx, yy, distance_field.reshape(xx.shape)
