from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

business_info_store = Chroma(
    collection_name="business_info",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

def get_business_info():
    # Get all docs with type 'business_info'
    docs = business_info_store.get(where={"type": "business_info"})
    # Extract all insights as a list
    if docs and docs['documents']:
        return docs['documents']  # list of strings
    return []