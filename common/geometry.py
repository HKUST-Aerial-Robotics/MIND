import numpy as np


def get_mahalanobis_distances(points, mean, covariance):
    vectors = points - mean
    inv_covariance = np.linalg.inv(covariance)
    left_term = np.dot(vectors, inv_covariance)
    distances = np.sqrt(np.sum(left_term * vectors, axis=1))
    return distances


def get_point_mean_distances(points, mean):
    vectors = points - mean
    distances = np.sqrt(np.sum(vectors * vectors, axis=1))
    return distances


def get_point_line_distance(points, line_start, line_end):
    # Calculate distance from a point to a line segment
    line_vec = line_end - line_start
    line_len_sq = np.dot(line_vec, line_vec)
    points_vec = points - line_start
    t = np.dot(points_vec, line_vec) / line_len_sq
    t = np.clip(t, 0, 1).reshape(-1, 1)
    projection = line_start + t * line_vec
    return np.linalg.norm(points - projection, axis=1)


def project_point_on_polyline(point, polyline):
    px, py = point
    sx, sy = polyline[:-1].T
    ex, ey = polyline[1:].T

    # Vectorized computation for projection
    dx, dy = ex - sx, ey - sy

    lengths_squared = dx ** 2 + dy ** 2

    assert np.all(lengths_squared != 0.0), "Polyline segments should not have zero lengths."

    t = np.clip(((px - sx) * dx + (py - sy) * dy) / lengths_squared, 0, 1)
    nearest_x = sx + t * dx
    nearest_y = sy + t * dy
    distances = np.sqrt((px - nearest_x) ** 2 + (py - nearest_y) ** 2)

    # Find the nearest segment
    nearest_index = np.argmin(distances)
    proj_pt = (nearest_x[nearest_index], nearest_y[nearest_index])

    # Cumulative distance sum up to the nearest segment (assuming the direction of the polyline is from start to end)
    cum_distance = (np.sum(np.sqrt(lengths_squared[:nearest_index])) +
                    np.sqrt(lengths_squared[nearest_index]) * t[nearest_index])

    # Compute heading (radians)
    heading = np.arctan2(dy[nearest_index], dx[nearest_index])

    return proj_pt, heading, cum_distance




