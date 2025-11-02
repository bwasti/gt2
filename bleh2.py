import gt
from gt import SignalConfig, ShardConfig

gt.gpu_workers(4)

data_parallel_config = SignalConfig(
    shard=ShardConfig(
        axis=0,
        workers=[0, 1, 2, 3]
    ),
    compile=False
)
gt.register_config('data_parallel', data_parallel_config)

with gt.signal.context('data_parallel'):
    x = gt.randn(1024, 1024)
    y = gt.randn(1024, 1024)
    for _ in range(100):
        for i in range(20):
            z = x @ y
            c = z.sum()
        print(c.data)

