#!/bin/bash

curl -X POST http://127.0.0.1:8080/rerank \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma3:1b", "query": "best web programming language", "top_n": 3, "documents": ["Python is great for data science", "JavaScript is popular for web development", "Rust provides memory safety without garbage collection"]}'
