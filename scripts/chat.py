from src.rag import ask

print("RAG Document Q&A Bot")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in {"exit", "quit"}:
        break
    if not question:
        continue
    try:
        answer, results = ask(question)
        print("\nAnswer:\n" + answer)
        print("\nRetrieved chunks:")
        for item in results:
            location = f"page {item['page']}" if item["page"] else f"chunk {item['chunk_id']}"
            print(f"- {item['source']} | {location} | score={item['score']:.3f}")
        print()
    except Exception as exc:
        print(f"\nError: {exc}\n")
