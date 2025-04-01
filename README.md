# RAG Project

A RAG (Retrieval-Augmented Generation) implementation using LlamaIndex and OpenAI.

## Setup

1. Clone the repository
```bash
git clone https://github.com/Dan-Cleary/rag_project.git
cd rag_project
```

2. Create a virtual environment and install dependencies
```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file and add your OpenAI API key
```bash
OPENAI_API_KEY=your_api_key_here
```

4. Add your documents to the `data/` directory

5. Run the project
```bash
python main.py
```

## Features
- Uses LlamaIndex for document indexing and retrieval
- Integrates with OpenAI's GPT-4 for generation
- Supports multiple document types through SimpleDirectoryReader 