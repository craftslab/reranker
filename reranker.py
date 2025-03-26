import os
import re
import requests
import uvicorn

from fastapi import FastAPI, Request

app = FastAPI()

ollama_model = os.environ.get('RERANK_MODEL', 'gemma3:1b')
ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

try:
    response = requests.get(f"{ollama_url}/api/tags")
    if response.status_code == 200:
        available_models = response.json().get('models', [])
        model_names = [m.get('name') for m in available_models]
        if ollama_model in model_names:
            print(f"Ollama model {ollama_model} is available")
        else:
            print(f"Warning: Model {ollama_model} not found in Ollama. Available models: {model_names}")
    else:
        print(f"Warning: Could not connect to Ollama API. Status code: {response.status_code}")
except Exception as e:
    print(f"Error connecting to Ollama: {e}")


def rerank_with_ollama(query, documents, ollama_model, ollama_url):
    results = []

    prompt_template = """
    I need you to rank the following documents by their relevance to the query.
    Return ONLY a relevance score between 0 and 1 for each document, where 1 is most relevant.

    Query: {query}

    Document: {document}

    Relevance score (0 to 1):
    """

    for doc in documents:
        prompt = prompt_template.format(query=query, document=doc)
        try:
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0
                    }
                }
            )
            if response.status_code == 200:
                completion = response.json().get('response', '')
                try:
                    score_text = completion.strip()
                    score_match = re.search(r'(\d+\.\d+|\d+)', score_text)
                    if score_match:
                        score = float(score_match.group(1))
                        score = max(0.0, min(1.0, score))
                    else:
                        score = 0.5
                except:
                    score = 0.5
                results.append({"document": doc, "score": score})
            else:
                results.append({"document": doc, "score": 0.5})
                print(f"Error with Ollama API: {response.status_code}")
        except Exception as e:
            results.append({"document": doc, "score": 0.5})
            print(f"Exception with Ollama API: {e}")

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    return sorted_results


@app.post("/v1/rerank")
async def rerank(request: Request):
    data = await request.json()

    query = data.get("query", "")
    documents = data.get("documents", [])
    top_n = data.get("top_n", len(documents))

    try:
        top_n = int(top_n)
        top_n = max(1, min(top_n, len(documents)))
    except (ValueError, TypeError):
        top_n = len(documents)

    current_model = data.get("model", ollama_model)
    sorted_results = rerank_with_ollama(query, documents, current_model, ollama_url)

    top_results = sorted_results[:top_n]

    return {
        "results": top_results,
        "model_used": current_model,
        "total_results": len(sorted_results),
        "returned_results": len(top_results)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
