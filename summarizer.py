from rag import retriever, llm


def summarize_topic(topic):
    """
    Creates a study-friendly summary from the uploaded notes.
    """

    documents = retriever.invoke(topic)

    if not documents:
        return "I couldn't find relevant information about this topic in your notes."

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are StudyMate AI, an academic study assistant.

Create a clear and useful summary of the topic below using
ONLY the information provided in the uploaded study notes.

Topic:
{topic}

Study Notes:
{context}

Format the answer exactly like this:

## 📌 Quick Summary
Give a short explanation of the topic.

## 🔑 Key Points
- Important point
- Important point
- Important point

## 🧠 Important Concepts
Explain the important concepts mentioned in the notes.

## 📝 Exam Revision
Give short revision points that a student can remember.

IMPORTANT:
- Use only the provided study notes.
- Do not add information that is not present in the notes.
- Keep the language easy to understand.
- Do not make the answer unnecessarily long.
"""

    response = llm.invoke(prompt)

    return response.content
if __name__ == "__main__":

    print("📝 StudyMate Summarizer")

    topic = input("\nEnter a topic to summarize: ")

    print("\n⏳ Creating summary...\n")

    result = summarize_topic(topic)

    print(result)