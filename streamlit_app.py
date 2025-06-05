import os
from dotenv import load_dotenv
import streamlit as st

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI

# Load environment variables
load_dotenv()

# Configure OpenAI model
Settings.llm = OpenAI(model="o3-mini", temperature=0)

@st.cache_resource
def load_index():
    """Load documents and create the vector index (cached)."""
    documents = SimpleDirectoryReader("data/").load_data()
    return VectorStoreIndex.from_documents(documents)


def main():
    st.title("RAG Demo")
    query = st.text_input("Enter your query", "give me two takeaways from each article")

    if st.button("Run Query"):
        index = load_index()
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        st.write(response)


if __name__ == "__main__":
    main()
