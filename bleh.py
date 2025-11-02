import gt
import time

while True:
   t0 = time.time()
   for _ in range(100):
       a = gt.randn(1024, 1024)
       b = gt.randn(1024, 1024)
       c = a @ b
       d = c.sum()
       __ = d.data
   t1 = time.time()
   print(t1 - t0)
