import gt

# Run some operations
x = gt.zeros(128, 128)
y = gt.zeros(128, 128)

for i in range(20):
    b = gt.zeros(128, 128)
    for j in range(50):
        b = x + y + b
    # Trigger computation
    _ = b[:1, :1]

# Get stats
print("\n=== Worker Statistics ===")
gt.debug.print_worker_stats()
