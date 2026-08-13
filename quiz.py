from rag import retriever, llm
import json
import re


def generate_quiz(topic, number_of_questions=5):
    """
    Generate MCQ quiz questions from the uploaded study notes.
    """

    # Retrieve relevant content
    documents = retriever.invoke(topic)

    if not documents:
        return []

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are StudyMate AI, an academic quiz generator.

Create a multiple-choice quiz using ONLY the study notes provided below.

Topic:
{topic}

Number of questions:
{number_of_questions}

Study Notes:
{context}

Return ONLY valid JSON.

Use exactly this format:

[
  {{
    "question": "Question here?",
    "options": {{
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D"
    }},
    "answer": "A",
    "explanation": "Short explanation of why A is correct."
  }}
]

Rules:
- Create exactly {number_of_questions} questions.
- Each question must have exactly 4 options.
- Only ONE option must be correct.
- The answer must be A, B, C, or D.
- Questions must be based only on the provided notes.
- Do not invent facts outside the notes.
- Keep questions suitable for students.
- Mix easy, medium and slightly challenging questions.
- Return ONLY JSON.
"""

    response = llm.invoke(prompt)

    text = response.content.strip()

    # Remove accidental markdown code fences
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*$",
        "",
        text
    )

    try:

        quiz = json.loads(text)

        if isinstance(quiz, list):
            return quiz

        return []

    except json.JSONDecodeError:

        # Try to extract JSON array
        match = re.search(
            r"\[.*\]",
            text,
            re.DOTALL
        )

        if match:

            try:
                return json.loads(match.group())

            except json.JSONDecodeError:
                return []

        return []


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("❓ StudyMate AI Quiz Generator")

    topic = input(
        "\nEnter a topic for the quiz: "
    )

    number = input(
        "Number of questions (default 5): "
    )

    try:
        number = int(number)

    except ValueError:
        number = 5

    print(
        "\n⏳ Generating quiz from your notes...\n"
    )

    quiz = generate_quiz(
        topic,
        number
    )

    if not quiz:

        print(
            "❌ Could not generate the quiz."
        )

    else:

        for i, question in enumerate(
            quiz,
            start=1
        ):

            print(
                f"\n{i}. {question['question']}"
            )

            for letter, option in question[
                "options"
            ].items():

                print(
                    f"   {letter}. {option}"
                )

            print(
                f"Answer: {question['answer']}"
            )

            print(
                f"Explanation: "
                f"{question['explanation']}"
            )