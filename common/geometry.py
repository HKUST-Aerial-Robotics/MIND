import numpy as np

def is_inside_ellipse(point, mean, cov):
    left_hand = (point - mean).T @ np.linalg.inv(cov) @ (point - mean)
    return left_hand <= 5.991


def ellipse_points(mean, cov, n=20):
    vals, vecs = np.linalg.eigh(cov)
    theta = np.linspace(0, 2 * np.pi, n)
    a, b = np.sqrt(np.abs(vals) * 5.991)
    ellipse_pts = np.array([a * np.cos(theta), b * np.sin(theta)])
    ellipse_pts_rotated = vecs @ ellipse_pts
    ellipse_pts_rotated[0, :] += mean[0]
    ellipse_pts_rotated[1, :] += mean[1]
    return ellipse_pts_rotated.T


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


def remove_close_points(points, min_dist):
    if len(points) < 2:
        return points

    filtered_points = [points[0]]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - filtered_points[-1]) > min_dist:
            filtered_points.append(points[i])
    return np.array(filtered_points)


# Vertices of a cube
def get_cube_vertices(x, y, z, dx, dy, dz):
    return [
        [x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z],
        [x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]
    ]


def rotate_vertex(vertex, radians):
    x, y, z = vertex
    x_prime = x * np.cos(radians) - y * np.sin(radians)
    y_prime = x * np.sin(radians) + y * np.cos(radians)
    return [x_prime, y_prime, z]


def get_vehicle_vertices(x, y, z, yaw, length, width, height):
    # get the left-bottom of axis-aligned bounding box of the vehicle
    x_lb = - length / 2
    y_lb = - width / 2
    axis_aligned_vertices = get_cube_vertices(x_lb, y_lb, z, length, width, height)
    # rotate the vertices
    rotated_vertices = [rotate_vertex(v, yaw) for v in axis_aligned_vertices]
    # translate the vertices to the center of the vehicle
    return [[v[0] + x, v[1] + y, v[2]] for v in rotated_vertices]


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


