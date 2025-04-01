from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# import modules:
# SimpleDirectoryReader loads docs from project, 
# VectorStoreIndex helps create an index by converting documents into vector embeddings.

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI

# Configure to use o3-mini by default
Settings.llm = OpenAI(model="o3-mini", temperature=0)

# 1. Load/read documents from the "data/" folder
documents = SimpleDirectoryReader("data/").load_data()

# 2. Create a vector store index from the documents. Convert documents into vector embeddings.
index = VectorStoreIndex.from_documents(documents)

# 3. Create a query engine
query_engine = index.as_query_engine()

# 4. Define your query
query = "give me two takeaways from each article"

# 5. Query the index and get the response 
#The query engine converts text into an embedding, retrieves the most relevant document chunks from the index, 
response = query_engine.query(query)

# 6. Print the generated answer
print(response)
