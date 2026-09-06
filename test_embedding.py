from langchain_ollama import OllamaEmbeddings

print("Loading embedding model...")

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

print("Generating test embedding...")

result = embeddings.embed_query(
    "I love the coffee at this restaurant."
)

print("\nEmbedding generated successfully!")

print("Vector length:", len(result))

print("First 10 values:")
print(result[:10])