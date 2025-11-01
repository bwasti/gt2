import gt
import time

x = gt.zeros(128, 128)
y = gt.zeros(128, 128)

for i in range(100):
    t0 = time.time()
    b = gt.zeros(128, 128)
    for i in range(50):
        b = x + y + b
    t1 = time.time()
    print(t1-t0, b[:1, :1])
