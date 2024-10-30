import numpy as np

class VehicleParam:
    def __init__(self, wb=3.0, max_spd=15.0, max_acc=6.0, max_str=np.deg2rad(45.0),
                 max_dstr=np.deg2rad(30.0)):
        self.wb = wb
        self.max_spd = max_spd
        self.max_acc = max_acc
        self.max_str = max_str
        self.max_dstr = max_dstr
        self.max_dec = -max_acc

    def load_from_json(self, json_data):
        self.wb = json_data['wheelbase']
        self.max_spd = json_data['max_speed']
        self.max_acc = json_data['max_accel']
        self.max_str = json_data['max_steer']
        self.max_dstr = json_data['max_steer_rate']
        self.max_dec = json_data['max_decel']


def kine_propagate(state: np.array, ctrl: np.array, dt: float, wb=2.5,
                   max_spd=20.0, max_steer=np.deg2rad(45.0),
                   max_acc=6.0, max_dec=-6.0):
    x, y, v, yaw = state
    a, delta = ctrl

    # input check
    a = np.clip(a, max_dec, max_acc)
    delta = np.clip(delta, -max_steer, max_steer)
    updated_state = np.array([x + v * np.cos(yaw) * dt,
                              y + v * np.sin(yaw) * dt,
                              v + a * dt,
                              yaw + v / wb * np.tan(delta) * dt])
    updated_state[2] = np.clip(updated_state[2], -max_spd, max_spd)
    return updated_state
