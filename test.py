from atari_wrappers import *
import time
env = make_atari('BreakoutNoFrameskip-v4')
env = wrap_deepmind(env,frame_stack=False,scale=True,frame_wrap = False)

print(time.time())