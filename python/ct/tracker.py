from . import libct

class Tracker:

    def __init__(self):
        self.tracker = libct.tracker_create()

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self.tracker is not None:
            libct.tracker_destroy(self.tracker)
            self.tracker = None

    def lower_bound(self):
        return libct.tracker_lower_bound(self.tracker)

    def run(self, max_iterations=1000):
        libct.tracker_run(self.tracker, max_iterations)
