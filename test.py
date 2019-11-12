from collections import deque
from util.ringbuf import RingBuf
from util.decorators import timethis

@timethis
def test():
    d = RingBuf(maxlen=100000)

    for i in range(200000):
        d.append(1)


if __name__ == "__main__":
    test()