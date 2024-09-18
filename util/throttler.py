from threading import Timer
class Throttler:
    def __init__(self, inteval):
        self.inteval=inteval
        self.task=None
        self.timer=None

    def set_target(self, task, args):
        self.task=task
        if self.timer is None or not self.timer.is_alive():
            self.timer= Timer(self.inteval, self.task, args)
            self.timer.start()
            return
        self.timer.cancel()
        self.timer= Timer(self.inteval, self.task, args)
        self.timer.start()

    def cancel(self):
        if self.timer is not None:
            self.timer.cancel()
            self.timer=None