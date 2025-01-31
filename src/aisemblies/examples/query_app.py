import json

import httpx


def try_crag():
    url = "http://localhost:8000/crag"
    questions = [
        "What is few-shot learning?",
        "How does prompt engineering differ from prompt tuning?",
    ]

    with httpx.Client() as client:
        for q in questions:
            response = client.post(url, json={"question": q}, timeout=3 * 60)
            print(f"\nQuestion: {q}")
            if response.status_code == 200:
                print("Response JSON:", response.json())
            else:
                print("Request failed with status:", response.status_code)


def stream_many():
    url = "http://localhost:8000/crag_many"
    questions = [
        "What is few-shot learning?",
        "How does prompt engineering differ from prompt tuning?",
    ]

    with httpx.stream("POST", url, json=questions, timeout=3 * 60) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            question = data.get("question")
            answer = data.get("answer")
            print(f"Question: {question}")
            print(f"Answer: {answer}\n")


if __name__ == "__main__":
    stream_many()
