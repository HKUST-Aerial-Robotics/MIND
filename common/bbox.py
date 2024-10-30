class BBox:
    def __init__(self, l, w, h):
        self.l = l
        self.w = w
        self.h = h

    def load_from_json(self, json_data):
        self.l = json_data['length']
        self.w = json_data['width']
        self.h = json_data['height']


class PedestrianBBox(BBox):
    def __init__(self):
        super(PedestrianBBox, self).__init__(0.5, 0.75, 1.8)


class CyclistBBox(BBox):
    def __init__(self):
        super(CyclistBBox, self).__init__(1.5, 0.75, 1.5)


class VehicleBBox(BBox):
    def __init__(self):
        super(VehicleBBox, self).__init__(4.5, 2.0, 1.5)


class BusBBox(BBox):
    def __init__(self):
        super(BusBBox, self).__init__(7.0, 2.1, 2.25)


class UnknownBBox(BBox):
    def __init__(self):
        super(UnknownBBox, self).__init__(1.0, 1.0, 1.0)
