from pathlib import Path
import sys
import re
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm

POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_FILE = ROOT / "outputs" / "policy_rag_results.csv"

STOP_WORDS = {
    "the", "a", "an", "is", "of", "to", "for", "in",
    "on", "how", "what", "when", "who", "are", "do",
    "does", "i", "my", "be", "can", "after", "from",
    "it", "or", "and", "at", "by", "which", "why"
}


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return {w for w in words if w not in STOP_WORDS}


def load_policies():
    documents = {}

    for file in sorted(POLICY_DIR.glob("*.txt")):
        documents[file.name] = file.read_text(encoding="utf-8")

    return documents


def split_into_chunks(documents):
    chunks = []

    for filename, content in documents.items():
        sentences = re.split(r"(?<=[.!?])\s+", content)

        for sentence in sentences:
            sentence = sentence.strip()

            if sentence:
                chunks.append({
                    "document": filename,
                    "text": sentence
                })

    return chunks


def retrieve_policy(question, chunks, top_k=3):
    query_words = tokenize(question)
    scored_chunks = []

    for chunk in chunks:
        chunk_words = tokenize(chunk["text"])
        overlap = query_words.intersection(chunk_words)

        score = len(overlap)

        if score > 0:
            scored_chunks.append({
                **chunk,
                "score": score
            })

    scored_chunks.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_chunks[:top_k]


def generate_answer(question, retrieved):
    if not retrieved or retrieved[0]["score"] < 2:
        return "I could not find that information in the policy documents."

    context = "\n".join(
        f"[{item['document']}] {item['text']}"
        for item in retrieved
    )

    messages = [
        {
            "role": "system",
            "content": "POLICY ASSISTANT: Answer only using the supplied policy context. Do not invent facts."
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"
        }
    ]

    response = mock_llm.chat(messages)

    return response.choices[0].message.content


def evaluate():
    documents = load_policies()
    chunks = split_into_chunks(documents)

    evaluation = pd.read_csv(EVAL_FILE)

    results = []

    for _, row in evaluation.iterrows():
        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        retrieved = retrieve_policy(question, chunks)

        answer = generate_answer(question, retrieved)

        if retrieved:
            retrieved_doc = retrieved[0]["document"]
        else:
            retrieved_doc = "NONE"

        if expected_doc == "NONE":
            correct_doc = retrieved_doc == "NONE"
            correct_answer = (
                "I could not find that information"
                in answer
            )
        else:
            correct_doc = expected_doc in [
                item["document"] for item in retrieved
            ]

            fact_words = tokenize(expected_fact)
            answer_words = tokenize(answer)

            correct_answer = (
                len(fact_words.intersection(answer_words)) > 0
            )

        results.append({
            "question": question,
            "expected_doc": expected_doc,
            "retrieved_doc": retrieved_doc,
            "answer": answer,
            "correct_doc": correct_doc,
            "correct_answer": correct_answer
        })

    result_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\nPOLICY RAG EVALUATION")
    print("--------------------")
    print("Total questions:", len(result_df))
    print(
        "Correct document retrieval:",
        result_df["correct_doc"].sum()
    )
    print(
        "Correct answers:",
        result_df["correct_answer"].sum()
    )
    print(
        "Document accuracy:",
        round(result_df["correct_doc"].mean() * 100, 2),
        "%"
    )
    print(
        "Answer accuracy:",
        round(result_df["correct_answer"].mean() * 100, 2),
        "%"
    )

    print("\nDETAILED RESULTS:")
    print(
        result_df[
            ["question", "retrieved_doc", "correct_doc", "correct_answer"]
        ].to_string(index=False)
    )

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    evaluate()