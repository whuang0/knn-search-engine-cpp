# High-Performance k-NN Search Engine

A cache-efficient k-d tree with parallel construction and query execution,
achieving near-linear scaling across CPU cores on datasets exceeding 1M points.

## Performance

Benchmarked on 1M training points, 10 dimensions, 10k queries, k=5:

| Threads | Search Time | Speedup |
|---------|-------------|---------|
| 1       | 14,640ms    | 1x      |
| 2       | 10,013ms    | 1.46x   |
| 4       | 3,214ms     | 4.55x   |
| 8       | 1,838ms     | 7.96x   |

## Architecture

**k-d tree construction** uses median partitioning via `nth_element` (O(n) vs O(n log n))
with round-robin dimension cycling. Above 100k points, subtrees are built in parallel
using `std::async`.

**k-NN search** maintains a max-heap of size k during traversal. The hyperplane
pruning condition eliminates subtrees that cannot contain closer neighbors,
avoiding sqrt by comparing squared distances throughout.

**Batch query execution** splits queries into per-thread chunks. Each thread
searches independently since the tree is read-only after construction.

**Named pipe IPC** supports a separate worker process mode for streaming
single queries from external processes.

## Build & Run

```bash
make
```

Generate test data:

```bash
python3 scripts/training_data.py 1000000 10 0
python3 scripts/query_file.py 10000 10 0 5
```

Run with N threads:

```bash
./knn_search <threads> <training.dat> <query.dat> <results.bin>
```

## Implementation Notes

- Squared Euclidean distance avoids sqrt during search hot path
- Worker process caches results by query ID to avoid redundant searches
- Memory-mapped I/O for training and result files
