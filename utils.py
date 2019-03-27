class RunningAverage:
    def __init__(self):
        self.counts = 0
        self.total_sum = 0
        self.avg = 0

    def reset(self):
        self.counts = 0
        self.total_sum = 0

    def update(self, val):
        self.total_sum += val
        self.counts += 1
        self.avg = self.total_sum / self.counts
