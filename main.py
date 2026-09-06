from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from vector import retriever, vector_store


# ============================================================
# 1. LOAD LLM
# ============================================================

model = OllamaLLM(
    model="llama3.2"
)


# ============================================================
# 2. PROMPT
# ============================================================

template = """
You are an intelligent AI assistant for a coffee restaurant.

You have access to:

1. Restaurant menu information
2. Customer reviews

IMPORTANT RULES:

- Answer ONLY using the retrieved information.
- Do not invent products, reviews, ratings, prices, or customer opinions.
- If the retrieved information does not contain the answer,
  clearly say that the available information does not provide
  enough information.
- When discussing customer opinions, distinguish between
  individual reviews and general patterns.
- If several reviews express similar opinions, summarize the
  common pattern.
- Keep the answer clear and useful.

Retrieved Information:

{context}

Customer Question:

{question}

Answer:
"""


prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def format_documents(documents):

    if not documents:
        return "No relevant information was retrieved."

    formatted = []

    for i, doc in enumerate(documents, 1):

        source = doc.metadata.get(
            "source",
            "unknown"
        )

        if source == "reviews_pdf":

            review_id = doc.metadata.get(
                "review_id",
                "unknown"
            )

            product = doc.metadata.get(
                "product",
                "unknown"
            )

            rating = doc.metadata.get(
                "rating",
                "unknown"
            )

            sentiment = doc.metadata.get(
                "sentiment",
                "unknown"
            )

            formatted.append(
                f"""
--- Review {i} ---

Review ID: {review_id}
Product: {product}
Rating: {rating}/5
Sentiment: {sentiment}

{doc.page_content}
"""
            )

        else:

            formatted.append(
                f"""
--- Menu Item {i} ---

{doc.page_content}
"""
            )

    return "\n".join(formatted)


# ============================================================
# 4. DETECT QUERY TYPE
# ============================================================

def get_query_type(question):

    q = question.lower()

    # --------------------------------------------------------
    # Positive review questions
    # --------------------------------------------------------

    positive_keywords = [
        "positive review",
        "positive reviews",
        "good reviews",
        "good review",
        "liked",
        "like",
        "love",
        "favorite",
        "favourite",
        "best reviews",
        "what do customers like",
        "what customers like",
        "what customers love"
    ]


    # --------------------------------------------------------
    # Negative review questions
    # --------------------------------------------------------

    negative_keywords = [
        "negative review",
        "negative reviews",
        "bad reviews",
        "bad review",
        "complaint",
        "complaints",
        "problem",
        "problems",
        "dislike",
        "disliked",
        "hate",
        "common complaints",
        "what customers dislike",
        "what customers complain"
    ]


    for keyword in positive_keywords:

        if keyword in q:
            return "positive"


    for keyword in negative_keywords:

        if keyword in q:
            return "negative"


    return "general"


# ============================================================
# 5. RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(question):

    query_type = get_query_type(question)


    # ========================================================
    # POSITIVE REVIEWS
    # ========================================================

    if query_type == "positive":

        print("\n[Retrieval mode: POSITIVE REVIEWS]")

        documents = vector_store.similarity_search(

            question,

            k=10,

            filter={
                "sentiment": "Positive"
            }
        )

        return documents


    # ========================================================
    # NEGATIVE REVIEWS
    # ========================================================

    elif query_type == "negative":

        print("\n[Retrieval mode: NEGATIVE REVIEWS]")

        documents = vector_store.similarity_search(

            question,

            k=10,

            filter={
                "sentiment": "Negative"
            }
        )

        return documents


    # ========================================================
    # GENERAL SEARCH
    # ========================================================

    else:

        print("\n[Retrieval mode: GENERAL SEARCH]")

        documents = retriever.invoke(question)

        return documents


# ============================================================
# 6. MAIN CHAT LOOP
# ============================================================

print("\n")
print("==============================================")
print("     COFFEE RESTAURANT RAG AI ASSISTANT")
print("==============================================")

print("\nLLM       : llama3.2")
print("Embedding : mxbai-embed-large")
print("Database  : ChromaDB")

print("\nAsk questions about:")
print("- Menu")
print("- Products")
print("- Customer reviews")
print("- Positive reviews")
print("- Negative reviews")
print("- Common complaints")
print("- Customer opinions")

print("\nType 'q' to quit.")


while True:

    print("\n----------------------------------------------")

    question = input(
        "Ask your question (q to quit): "
    ).strip()


    # ========================================================
    # EXIT
    # ========================================================

    if question.lower() == "q":
        print("\nGoodbye!")
        break


    if not question:
        continue


    # ========================================================
    # RETRIEVE
    # ========================================================

    try:

        retrieved_docs = retrieve_documents(
            question
        )

    except Exception as e:

        print("\nRetrieval error:")
        print(e)

        continue


    # ========================================================
    # SHOW RETRIEVAL COUNT
    # ========================================================

    print(
        f"\nRetrieved {len(retrieved_docs)} documents."
    )


    # ========================================================
    # CREATE CONTEXT
    # ========================================================

    context = format_documents(
        retrieved_docs
    )


    # ========================================================
    # SEND TO LLM
    # ========================================================

    try:

        result = chain.invoke({

            "context": context,

            "question": question
        })

    except Exception as e:

        print("\nLLM error:")
        print(e)

        continue


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    print("\nAnswer:")
    print(result)