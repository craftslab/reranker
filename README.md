# reranker

## Build

```bash
docker build -f Dockerfile -t craftslab/reranker:latest .
```

## Run

```bash
docker compose up -d
```

## Test

```bash
curl -X POST http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma3:1b", "query": "best web programming language", "top_n": 3, "documents": ["Python is great for data science", "JavaScript is popular for web development", "Rust provides memory safety without garbage collection"]}'
```

## Reference

- [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3#usage)
- [jina-reranker](https://jina.ai/reranker/)
- [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference/blob/main/README.md)
