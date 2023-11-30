import time


# TODO wrap in class for safety
# TODO start/stop using the 'with <item>:' context manager


times={}


def start(key):
    if not key in times:
        times[key]=[0.0,0.0]
    times[key][1]=time.time()


def stop(key):
    times[key][0]=time.time()-times[key][1]


def prtTimes():
    global times

    times=list(map(lambda k,v:(k,v[0]),times.keys(),times.values()))
    times.sort(key=lambda t:t[1],reverse=True)

    convertTime=lambda t:f"{t[0]} =".ljust(15)+f" {int(round(t[1]*1000))/1000}\n"
    times=list(map(convertTime,times))
    print("times:")
    print("".join(times))


def clrTimes():
    global times
    times={}
