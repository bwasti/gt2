#!/bin/bash
# Quick launcher for 8 GPU benchmark
# Usage: bash benchmarks/launch_8gpu_benchmark.sh

set -e

PORT=9000
NUM_GPUS=8

echo "=================================="
echo "GT Auto-Shard 8 GPU Benchmark"
echo "=================================="
echo ""

# Check if already running
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port $PORT is already in use. Killing existing processes..."
    pkill -f "gt.server" || true
    pkill -f "gt.worker" || true
    sleep 2
fi

echo "Step 1: Starting dispatcher on port $PORT..."
python -m gt.server -p $PORT &
DISPATCHER_PID=$!
echo "  Dispatcher PID: $DISPATCHER_PID"
sleep 2

echo ""
echo "Step 2: Starting $NUM_GPUS GPU workers..."
WORKER_PIDS=()
for i in $(seq 0 $((NUM_GPUS-1))); do
    echo "  Starting worker $i on GPU $i..."
    CUDA_VISIBLE_DEVICES=$i python -m gt.worker --host localhost -p $PORT &
    WORKER_PIDS+=($!)
done

echo ""
echo "Waiting for workers to connect..."
sleep 5

echo ""
echo "✓ System ready!"
echo ""
echo "=================================="
echo "Running Benchmarks"
echo "=================================="

# Run benchmark comparing 1 GPU vs 8 GPUs
GT_VERBOSE=1 python benchmarks/auto_shard_benchmark.py --gpus 1 8 --port $PORT

BENCHMARK_EXIT=$?

echo ""
echo "=================================="
echo "Cleaning Up"
echo "=================================="

# Kill all processes
echo "Stopping workers..."
for pid in "${WORKER_PIDS[@]}"; do
    kill $pid 2>/dev/null || true
done

echo "Stopping dispatcher..."
kill $DISPATCHER_PID 2>/dev/null || true

# Wait for cleanup
sleep 2

if [ $BENCHMARK_EXIT -eq 0 ]; then
    echo ""
    echo "✓ Benchmark completed successfully!"
else
    echo ""
    echo "⚠️  Benchmark failed with exit code $BENCHMARK_EXIT"
fi

exit $BENCHMARK_EXIT
