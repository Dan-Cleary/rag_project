from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI

# Configure to use GPT-4 by default
Settings.llm = OpenAI(model="gpt-4o", temperature=0)

# 1. Load documents from the "data/" folder
documents = SimpleDirectoryReader("data/").load_data()

# 2. Create a vector store index from the documents
index = VectorStoreIndex.from_documents(documents)

# 3. Create a query engine
query_engine = index.as_query_engine()

# 4. Define your query
query = "give me two takeaways from each article"

# 5. Query the index and get the response
response = query_engine.query(query)

# 6. Print the generated answer
print(response)
