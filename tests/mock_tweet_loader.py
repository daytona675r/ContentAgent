import json
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# Initialize Chroma vectorstore
chroma_store = Chroma(
    collection_name="tweet_history",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="../chroma_db"
)

# Load mock tweet data
with open("mock_tweet_history.json", "r") as f:
    tweet_data = json.load(f)

# Prepare and add documents
documents = []
for tweet in tweet_data:
        metadata = {
            "tweet_id": tweet["tweet_id"],
            "theme": tweet["theme"],
            "tone": tweet["tone"],
            "style": tweet["style"],
            "likes": tweet["engagement"]["likes"],
            "retweets": tweet["engagement"]["retweets"],
            "replies": tweet["engagement"]["replies"],
            "date": tweet["date"],
            "time_posted": tweet["time_posted"]
        }
        doc = Document(page_content=tweet["text"], metadata=metadata)
        documents.append(doc)

# Add to vectorstore
chroma_store.add_documents(documents)
chroma_store.persist()

print(f"✅ Loaded {len(documents)} top-performing tweets into Chroma DB.")
