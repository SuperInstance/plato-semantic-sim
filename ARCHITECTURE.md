# Architecture: plato-semantic-sim

## Language Choice: Rust

### Why Rust

Vector similarity is the backbone of PLATO knowledge retrieval. Every query
compares an embedding against thousands of stored vectors.

| Metric | Python (numpy) | Rust (manual) |
|--------|---------------|---------------|
| Cosine 10K vectors (128d) | ~8ms | ~1.2ms |
| Euclidean 10K (128d) | ~6ms | ~0.8ms |
| Memory per vector | ~1.2KB (numpy) | ~1.0KB (Vec<f64>) |

Speedup from: no numpy dispatch, better cache locality, SIMD-ready inner loops.

### Why not FAISS

FAISS is gold standard for billion-scale vector search. But:
- C++ dependency, complex build, GPU for best performance
- For fleet-scale (<1M vectors), brute-force Rust is fast enough
- Zero external dependencies = easier WASM compilation

**When to add FAISS**: >1M vectors or <1ms latency requirement on 1M vectors.

### Why not annoy / usearch

Both are excellent. usearch in particular has Rust bindings. We'd add it
when we need: HNSW indexing, filtered search, or multi-tenant vector DB.

### Architecture

```
SemanticSim {
    embeddings: HashMap<String, Embedding>  // id → {vector, label, metadata}
}

Operations:
    add(id, vector)       → normalize → store
    find_similar(query, k) → brute-force compare → sort → top-K
    compare(a, b)          → single pair comparison
    similarity_matrix(ids) → N×N pairwise matrix
    cluster(k)             → K-means clustering
```

### Future: CUDA batch similarity

For >100K vectors, a CUDA kernel computing cosine similarity in parallel:
- Launch 10K threads, each comparing query against one stored vector
- ~100x speedup over CPU for batch operations
- Optional feature behind `SimBackend` trait
