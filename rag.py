import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from retriever import retriever

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


def ask_question(question):
    documents = retriever.invoke(question)

    if not documents:
        return "I couldn't find relevant information in your uploaded notes."

    context = "\n\n".join(
        document.page_content for document in documents
    )

    prompt = f"""
You are StudyMate AI, a study assistant.

Answer the student's question using ONLY the information
provided in the context below.

If the answer is not present in the context, say:
"I couldn't find that information in your uploaded notes."

Context:
{context}

Question:
{question}

Answer clearly and in a student-friendly way:
"""

    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":
    print("🎓 StudyMate AI is ready!")

    question = input("\nAsk a question: ")

    answer = ask_question(question)

    print("\n🤖 Answer:")
    print(answer)