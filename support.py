import time


class Timer():
    times = {}

    def __init__(self, key):
        self.key = key
        self.start = None
        if not self.key in Timer.times:
            Timer.times[self.key] = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args, **kwds):
        Timer.times[self.key] += time.time()-self.start

    @staticmethod
    def prtTimes():
        times = list(Timer.times.items())
        times.sort(key=lambda t: t[1], reverse=True)

        times = list(map(lambda t: f'{t[0]} ='.ljust(15)+f'{t[1]:.4}\n', times))
        print('times:')
        print(''.join(times))

    @staticmethod
    def clrTimes():
        Timer.times = {}
