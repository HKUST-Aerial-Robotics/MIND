import numpy as np


class ControlPotential:
    def __init__(self, weight):
        self.weight = weight

    def get_potential(self, control):
        return control.T.dot(self.weight).dot(control)

    def get_gradient(self, control):
        return 2.0 * self.weight.dot(control)

    def get_hessian(self, control):
        return 2.0 * self.weight


class StateConstraint:
    def __init__(self, weight, lower_bound, upper_bound):
        self.weight = weight
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_potential(self, state):
        diff = np.maximum(state - self.upper_bound, 0) + np.maximum(self.lower_bound - state, 0)
        return diff.T.dot(self.weight).dot(diff)

    def get_gradient(self, state):
        ret_grad = np.zeros_like(state)
        for i in range(len(state)):
            if state[i] > self.upper_bound[i]:
                ret_grad[i] = 2.0 * self.weight[i, i] * (state[i] - self.upper_bound[i])
            elif state[i] < self.lower_bound[i]:
                ret_grad[i] = 2.0 * self.weight[i, i] * (state[i] - self.lower_bound[i])
        return ret_grad

    def get_hessian(self, state):
        ret_hessian = np.zeros((state.shape[0], state.shape[0]))
        for i in range(len(state)):
            if state[i] > self.upper_bound[i] or state[i] < self.lower_bound[i]:
                ret_hessian[i, i] = 2.0 * self.weight[i, i]
        return ret_hessian


class StatePotential:
    def __init__(self, weight, des_state):
        self.des_state = des_state
        self.weight = weight

    def get_potential(self, state):
        diff = state - self.des_state
        return diff.T.dot(self.weight).dot(diff)

    def get_gradient(self, state):
        diff = state - self.des_state
        return 2.0 * self.weight.dot(diff)

    def get_hessian(self, state):
        return 2.0 * self.weight


class PotentialField:
    def __init__(self, field_offset, resolution, xx, yy, cost_field):
        self.offset = field_offset
        self.res = resolution
        self.xx = xx
        self.yy = yy
        self.limits = (np.min(xx), np.max(xx), np.min(yy), np.max(yy))
        self.cost_field = cost_field
        self.cache = {}

    def get_potential(self, state):
        query_pos = state[:2]
        x_idx, y_idx = self._get_idx_from_pos(query_pos)
        grid_ori = self._get_grid_ori(x_idx, y_idx)
        smooth_local_grid = self._get_smooth_local_grid(x_idx, y_idx)
        u, v = self._get_uv(query_pos, grid_ori)
        cost = self._quadratic_interpolation(u, v, smooth_local_grid)
        return cost

    def get_gradient(self, state):
        query_pos = state[:2]
        x_idx, y_idx = self._get_idx_from_pos(query_pos)
        grid_ori = self._get_grid_ori(x_idx, y_idx)
        smooth_local_grid = self._get_smooth_local_grid(x_idx, y_idx)
        u, v = self._get_uv(query_pos, grid_ori)
        ret_gradient = np.zeros_like(state)
        ret_gradient[:2] = self._compute_gradient(u, v, smooth_local_grid)
        return ret_gradient

    def get_hessian(self, state):
        query_pos = state[:2]
        x_idx, y_idx = self._get_idx_from_pos(query_pos)
        grid_ori = self._get_grid_ori(x_idx, y_idx)
        smooth_local_grid = self._get_smooth_local_grid(x_idx, y_idx)
        u, v = self._get_uv(query_pos, grid_ori)
        ret_hessian = np.zeros((state.shape[0], state.shape[0]))
        ret_hessian[:2, :2] = self._compute_hessian(u, v, smooth_local_grid)
        return ret_hessian

    def get_limits(self):
        return self.limits

    def _get_idx_from_pos(self, query_pos):
        # clamp the query position to the limits
        x_idx = round((query_pos[0] - self.offset[0]) / self.res)
        y_idx = round((query_pos[1] - self.offset[1]) / self.res)
        x_idx = max(0, min(self.cost_field.shape[1] - 1, x_idx))
        y_idx = max(0, min(self.cost_field.shape[0] - 1, y_idx))
        return x_idx, y_idx

    def _retrieve_from_cache(self, x_idx, y_idx):
        cache_key = (x_idx, y_idx)
        if cache_key in self.cache:
            return True
        return False

    def _add_to_cache(self, x_idx, y_idx, value):
        cache_key = (x_idx, y_idx)
        self.cache[cache_key] = value

    def _get_smooth_local_grid(self, x_idx, y_idx):
        if self._retrieve_from_cache(x_idx, y_idx):
            return self.cache[(x_idx, y_idx)]

        local_grid = np.zeros((3, 3))
        if x_idx == 0 and y_idx == 0:
            local_grid[1:, 1:] = self.cost_field[:2, :2]
        elif x_idx == 0 and y_idx == self.cost_field.shape[0] - 1:
            local_grid[1:, :2] = self.cost_field[-2:, :2]
        elif x_idx == self.cost_field.shape[1] - 1 and y_idx == 0:
            local_grid[:2, 1:] = self.cost_field[:2, -2:]
        elif x_idx == self.cost_field.shape[1] - 1 and y_idx == self.cost_field.shape[0] - 1:
            local_grid[:2, :2] = self.cost_field[-2:, -2:]
        elif x_idx == 0:
            local_grid[:, :2] = self.cost_field[y_idx - 1:y_idx + 2, :2]
        elif x_idx == self.cost_field.shape[1] - 1:
            local_grid[:, 1:] = self.cost_field[y_idx - 1:y_idx + 2, -2:]
        elif y_idx == 0:
            local_grid[:2, :] = self.cost_field[:2, x_idx - 1:x_idx + 2]
        elif y_idx == self.cost_field.shape[0] - 1:
            local_grid[1:, :] = self.cost_field[-2:, x_idx - 1:x_idx + 2]
        else:
            local_grid = self.cost_field[y_idx - 1:y_idx + 2, x_idx - 1:x_idx + 2]

        smooth_local_grid = np.zeros((3, 3))
        smooth_local_grid[0, 0] = np.mean(local_grid[:2, :2])
        smooth_local_grid[0, 2] = np.mean(local_grid[:2, 1:])
        smooth_local_grid[2, 0] = np.mean(local_grid[1:, :2])
        smooth_local_grid[2, 2] = np.mean(local_grid[1:, 1:])
        smooth_local_grid[0, 1] = np.mean(local_grid[:2, 1])
        smooth_local_grid[1, 0] = np.mean(local_grid[1, :2])
        smooth_local_grid[1, 2] = np.mean(local_grid[1, 1:])
        smooth_local_grid[2, 1] = np.mean(local_grid[1:, 1])
        smooth_local_grid[1, 1] = np.mean(local_grid[1, 1])

        self._add_to_cache(x_idx, y_idx, smooth_local_grid)

        return smooth_local_grid

    def _get_grid_ori(self, x_idx, y_idx):
        return np.array([self.xx[y_idx, x_idx], self.yy[y_idx, x_idx]])

    def _get_uv(self, query_pos, grid_ori):
        u = (query_pos[0] - grid_ori[0]) / self.res + 0.5
        v = (query_pos[1] - grid_ori[1]) / self.res + 0.5
        return u, v

    def _quadratic_interpolation(self, u, v, grid):
        return (
                (1 - u) ** 2 * (1 - v) ** 2 * grid[0, 0] +
                (1 - u) ** 2 * 2.0 * (1 - v) * v * grid[1, 0] +
                (1 - u) ** 2 * v ** 2 * grid[2, 0] +
                2.0 * (1 - u) * u * (1 - v) ** 2 * grid[0, 1] +
                2.0 * (1 - u) * u * 2.0 * (1 - v) * v * grid[1, 1] +
                2.0 * (1 - u) * u * v ** 2 * grid[2, 1] +
                u ** 2 * (1 - v) ** 2 * grid[0, 2] +
                u ** 2 * 2.0 * (1 - v) * v * grid[1, 2] +
                u ** 2 * v ** 2 * grid[2, 2]
        )

    def _compute_gradient(self, u, v, grid):
        gradient_x = self._partial_derivative_x(u, v, grid)
        gradient_y = self._partial_derivative_y(u, v, grid)
        return np.array([gradient_x, gradient_y])

    def _partial_derivative_x(self, u, v, grid):
        return (
                1.0 / self.res * (
                (-2.0 + 2.0 * u) * (1.0 - v) ** 2 * grid[0, 0] +
                (-2.0 + 2.0 * u) * 2.0 * (1.0 - v) * v * grid[1, 0] +
                (-2.0 + 2.0 * u) * v ** 2 * grid[2, 0] +
                2.0 * (1.0 - 2.0 * u) * (1.0 - v) ** 2 * grid[0, 1] +
                2.0 * (1.0 - 2.0 * u) * 2.0 * (1.0 - v) * v * grid[1, 1] +
                2.0 * (1.0 - 2.0 * u) * v ** 2 * grid[2, 1] +
                u * 2.0 * (1.0 - v) ** 2 * grid[0, 2] +
                u * 2.0 * 2.0 * (1.0 - v) * v * grid[1, 2] +
                u * 2.0 * v ** 2 * grid[2, 2]
        )
        )

    def _partial_derivative_y(self, u, v, grid):
        return (
                1.0 / self.res * (
                (1.0 - u) ** 2 * (-2.0 + 2.0 * v) * grid[0, 0] +
                (1.0 - u) ** 2 * 2.0 * (1.0 - 2.0 * v) * grid[1, 0] +
                (1.0 - u) ** 2 * 2.0 * v * grid[2, 0] +
                2.0 * (1.0 - u) * u * (-2.0 + 2.0 * v) * grid[0, 1] +
                2.0 * (1.0 - u) * u * 2.0 * (1.0 - 2.0 * v) * grid[1, 1] +
                2.0 * (1.0 - u) * u * 2.0 * v * grid[2, 1] +
                u ** 2 * (-2.0 + 2.0 * v) * grid[0, 2] +
                u ** 2 * 2.0 * (1.0 - 2.0 * v) * grid[1, 2] +
                u ** 2 * 2.0 * v * grid[2, 2]
        )
        )

    def _compute_hessian(self, u, v, grid):
        d2f_dx2 = self._second_partial_derivative_x(u, v, grid)
        d2f_dy2 = self._second_partial_derivative_y(u, v, grid)
        d2f_dxdy = self._mixed_partial_derivative(u, v, grid)
        return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])

    def _second_partial_derivative_x(self, u, v, grid):
        return (
                1.0 / self.res ** 2 * (
                2.0 * (1.0 - v) ** 2 * grid[0, 0] +
                2.0 * (1.0 - v) * 2.0 * v * grid[1, 0] +
                2.0 * v ** 2 * grid[2, 0] +
                -4.0 * (1.0 - v) ** 2 * grid[0, 1] +
                -4.0 * (1.0 - v) * 2.0 * v * grid[1, 1] +
                -4.0 * v ** 2 * grid[2, 1] +
                2.0 * (1.0 - v) ** 2 * grid[0, 2] +
                2.0 * (1.0 - v) * 2.0 * v * grid[1, 2] +
                2.0 * v ** 2 * grid[2, 2])
        )

    def _second_partial_derivative_y(self, u, v, grid):
        return (
                1.0 / self.res ** 2 * (
                2.0 * (1.0 - u) ** 2 * grid[0, 0] +
                -4.0 * (1.0 - u) ** 2 * grid[1, 0] +
                2.0 * (1.0 - u) ** 2 * grid[2, 0] +
                2.0 * (1.0 - u) * 2.0 * u * grid[0, 1] +
                -4.0 * (1.0 - u) * 2.0 * u * grid[1, 1] +
                2.0 * (1.0 - u) * 2.0 * u * grid[2, 1] +
                2.0 * u ** 2 * grid[0, 2] +
                -4.0 * u ** 2 * grid[1, 2] +
                2.0 * u ** 2 * grid[2, 2])
        )

    def _mixed_partial_derivative(self, u, v, grid):
        return (
                1.0 / self.res ** 2 * (
                (-2.0 + 2.0 * u) * (-2.0 + 2.0 * v) * grid[0, 0] +
                (-2.0 + 2.0 * u) * 2.0 * (1.0 - 2.0 * v) * grid[1, 0] +
                (-2.0 + 2.0 * u) * 2.0 * v * grid[2, 0] +
                2.0 * (1.0 - 2.0 * u) * (-2.0 + 2.0 * v) * grid[0, 1] +
                2.0 * (1.0 - 2.0 * u) * 2.0 * (1.0 - 2.0 * v) * grid[1, 1] +
                2.0 * (1.0 - 2.0 * u) * 2.0 * v * grid[2, 1] +
                2.0 * u * (-2.0 + 2.0 * v) * grid[0, 2] +
                2.0 * u * 2.0 * (1.0 - 2.0 * v) * grid[1, 2] +
                2.0 * u * 2.0 * v * grid[2, 2]
        )
        )
