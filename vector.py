import os
import re
import shutil
import pandas as pd

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "coffee_restaurant_large_dataset.csv"
PDF_FILE = "coffee_restaurant_product_reviews_5000.pdf"

DB_LOCATION = "chroma_langchain_db"

COLLECTION_NAME = "coffee_restaurant"

# IMPORTANT:
# Set this to True ONLY when you want to completely rebuild
# the ChromaDB from CSV + PDF.
#
# After rebuilding successfully, change it to False.
RESET_DB = True


# ============================================================
# 1. RESET OLD DATABASE
# ============================================================

if RESET_DB and os.path.exists(DB_LOCATION):

    print("\nRemoving old ChromaDB...")

    shutil.rmtree(DB_LOCATION)

    print("Old ChromaDB removed successfully!")


# ============================================================
# 2. LOAD CSV MENU DATA
# ============================================================

print("\n======================================")
print("Loading restaurant menu CSV...")
print("======================================")

df = pd.read_csv(CSV_FILE)

print("CSV loaded successfully!")

print("\nCSV columns:")
print(df.columns.tolist())

print(f"\nNumber of menu items: {len(df)}")


# ============================================================
# 3. CREATE DOCUMENTS FROM CSV
# ============================================================

print("\nCreating menu documents...")

csv_documents = []
csv_ids = []


for index, row in df.iterrows():

    content = f"""
Restaurant Menu Item

Item ID: {row['item_id']}
Item Name: {row['item_name']}
Category: {row['category']}
Size: {row['size']}
Temperature: {row['temperature']}
Description: {row['description']}
Price: INR {row['price_inr']}
Rating: {row['rating']}
Calories: {row['calories']}
Availability: {row['availability']}
"""

    document = Document(
        page_content=content.strip(),

        metadata={
            "source": "menu_csv",
            "item_id": str(row["item_id"]),
            "item_name": str(row["item_name"]),
            "category": str(row["category"])
        }
    )

    csv_documents.append(document)

    csv_ids.append(f"menu_{index}")


print(f"Menu documents created: {len(csv_documents)}")


# ============================================================
# 4. LOAD PDF
# ============================================================

print("\n======================================")
print("Loading customer review PDF...")
print("======================================")

reader = PdfReader(PDF_FILE)

print(f"PDF pages: {len(reader.pages)}")


# ============================================================
# 5. EXTRACT COMPLETE PDF TEXT
# ============================================================

print("\nExtracting PDF text...")

full_text = ""

for page in reader.pages:

    text = page.extract_text()

    if text:
        full_text += "\n" + text


print("PDF text extraction completed!")

print(f"Extracted characters: {len(full_text)}")


# ============================================================
# 6. NORMALIZE TEXT
# ============================================================

# Replace Windows line endings
full_text = full_text.replace("\r\n", "\n")

# Replace multiple spaces/tabs
full_text = re.sub(r"[ \t]+", " ", full_text)

# Keep newlines


# ============================================================
# 7. SPLIT PDF INTO INDIVIDUAL REVIEWS
# ============================================================

print("\n======================================")
print("Parsing individual reviews...")
print("======================================")


# Every review starts with something like:
#
# Review 00001
# Review 00002
# Review 03951
#
# We split whenever a new Review number appears.

review_pattern = r"(?=Review\s+\d+)"

review_blocks = re.split(review_pattern, full_text)


# Remove empty blocks
review_blocks = [
    block.strip()
    for block in review_blocks
    if block.strip()
]


print(f"Review blocks detected: {len(review_blocks)}")


# ============================================================
# 8. PARSE EACH REVIEW
# ============================================================

review_documents = []
review_ids = []

successful_reviews = 0


for block in review_blocks:

    # --------------------------------------------------------
    # Review ID
    # --------------------------------------------------------

    review_id_match = re.search(
        r"Review\s+(\d+)",
        block,
        re.IGNORECASE
    )

    if not review_id_match:
        continue

    review_id = review_id_match.group(1)


    # --------------------------------------------------------
    # Product
    # --------------------------------------------------------

    product_match = re.search(
        r"Product:\s*(.*?)(?:\n|Rating:)",
        block,
        re.IGNORECASE
    )

    product = (
        product_match.group(1).strip()
        if product_match
        else "Unknown"
    )


    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    rating_match = re.search(
        r"Rating:\s*(\d+(?:\.\d+)?)\s*/\s*5",
        block,
        re.IGNORECASE
    )

    rating = (
        rating_match.group(1)
        if rating_match
        else "Unknown"
    )


    # --------------------------------------------------------
    # Sentiment
    # --------------------------------------------------------

    sentiment_match = re.search(
        r"Sentiment:\s*(.*?)(?:\n|Price Paid:)",
        block,
        re.IGNORECASE
    )

    sentiment = (
        sentiment_match.group(1).strip()
        if sentiment_match
        else "Unknown"
    )


    # --------------------------------------------------------
    # Price Paid
    # --------------------------------------------------------

    price_match = re.search(
        r"Price Paid:\s*(?:INR|₹)?\s*(\d+(?:\.\d+)?)",
        block,
        re.IGNORECASE
    )

    price_paid = (
        price_match.group(1)
        if price_match
        else "Unknown"
    )


    # --------------------------------------------------------
    # Review Text
    # --------------------------------------------------------

    review_match = re.search(
        r"Review:\s*(.*)",
        block,
        re.IGNORECASE | re.DOTALL
    )

    if review_match:

        review_text = review_match.group(1).strip()

    else:

        review_text = ""


    # --------------------------------------------------------
    # Clean review text
    # --------------------------------------------------------

    review_text = re.sub(
        r"\s+",
        " ",
        review_text
    ).strip()


    # --------------------------------------------------------
    # Ignore invalid reviews
    # --------------------------------------------------------

    if not review_text:
        continue


    # --------------------------------------------------------
    # Create readable document
    # --------------------------------------------------------

    content = f"""
Customer Review

Review ID: {review_id}
Product: {product}
Rating: {rating}/5
Sentiment: {sentiment}
Price Paid: INR {price_paid}

Customer Review:
{review_text}
"""


    document = Document(

        page_content=content.strip(),

        metadata={

            "source": "reviews_pdf",

            "review_id": str(review_id),

            "product": str(product),

            "rating": str(rating),

            "sentiment": str(sentiment),

            "price_paid": str(price_paid)
        }
    )


    review_documents.append(document)

    review_ids.append(f"review_{review_id}")

    successful_reviews += 1


# ============================================================
# 9. SHOW REVIEW STATISTICS
# ============================================================

print("\n======================================")
print("Review parsing completed")
print("======================================")

print(f"Successfully parsed reviews: {successful_reviews}")


# Count sentiments

positive_count = sum(
    1
    for doc in review_documents
    if doc.metadata["sentiment"].lower() == "positive"
)

negative_count = sum(
    1
    for doc in review_documents
    if doc.metadata["sentiment"].lower() == "negative"
)

neutral_count = sum(
    1
    for doc in review_documents
    if doc.metadata["sentiment"].lower() == "neutral"
)


print(f"Positive reviews: {positive_count}")
print(f"Negative reviews: {negative_count}")
print(f"Neutral reviews: {neutral_count}")


# ============================================================
# 10. COMBINE MENU + REVIEWS
# ============================================================

all_documents = csv_documents + review_documents

all_ids = csv_ids + review_ids


print("\n======================================")
print("Final Dataset")
print("======================================")

print(f"Menu documents   : {len(csv_documents)}")
print(f"Review documents : {len(review_documents)}")
print(f"Total documents  : {len(all_documents)}")


# ============================================================
# 11. LOAD EMBEDDING MODEL
# ============================================================

print("\n======================================")
print("Loading embedding model...")
print("======================================")

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

print("Embedding model loaded!")


# ============================================================
# 12. CREATE CHROMADB
# ============================================================

print("\n======================================")
print("Creating ChromaDB...")
print("======================================")


vector_store = Chroma(

    collection_name=COLLECTION_NAME,

    persist_directory=DB_LOCATION,

    embedding_function=embeddings
)


# ============================================================
# 13. ADD DOCUMENTS
# ============================================================

if RESET_DB:

    print("\nGenerating embeddings...")
    print(
        "This can take some time because "
        "there are many documents."
    )
    print("Adding documents in batches...")

    BATCH_SIZE = 25

    total = len(all_documents)

    for start in range(0, total, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total)

        batch_documents = all_documents[start:end]
        batch_ids = all_ids[start:end]

        print(
            f"Embedding documents "
            f"{start + 1} to {end} "
            f"of {total}"
        )

        vector_store.add_documents(
            documents=batch_documents,
            ids=batch_ids
        )

    print("\n======================================")
    print("ChromaDB created successfully!")
    print("======================================")

else:

    print("\nUsing existing ChromaDB.")
    print("\n======================================")
    print("ChromaDB created successfully!")
    print("======================================")


# ============================================================
# 14. CREATE GENERAL RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(

    search_kwargs={
        "k": 8
    }
)


# ============================================================
# 15. EXPORT VECTOR STORE
# ============================================================

# main.py can use vector_store directly for
# metadata filtering.

print("\nRetriever is ready!")

print("\nVector database setup completed!")