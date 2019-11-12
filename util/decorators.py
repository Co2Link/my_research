import time
from functools import wraps

def timethis(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        end = time.time()
        print("{} {:.3f}".format(func.__name__,end-start))
        return result
    return wrapper