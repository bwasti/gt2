import gt
import numpy as np

# gt.connect('localhost:12345') ## THIS IS OPTIONAL! if we don't connect the server is spun up automatically
a = gt.tensor([1, 2, 3, 4], dtype='float32')
b = gt.tensor([1, 2, 3, 4], dtype='float32')
c = a - b
data = c.data.numpy()

print("Result:", data)

expected = np.zeros((4,), dtype='float32')
np.testing.assert_array_equal(data, expected)

print("✓ Test passed!")
