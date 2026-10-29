# UMSI Policy RAG Assistant

A Retrieval-Augmented Generation (RAG) system designed to answer student questions about University of Michigan School of Information (UMSI) course registration, degree requirements, and academic policies.

This project uses LangChain, OpenAI embeddings, and ChromaDB to search through official student handbooks and policy documents to provide accurate, context-aware answers.

## Project Structure

* `/data/raw_documents/`: Contains the official UMSI PDF policy documents.
* `/data/chroma_db/`: The local vector database containing our document embeddings.
* `/src/`: Contains the Python source code for the RAG pipeline.
* `.env`: Local environment file storing the OpenAI API key.
* `requirements.txt`: Python dependencies required to run the project.

## Setup Instructions

To run the pipeline locally on your machine:

1. **Clone the repository**
   Download this repository to your local machine using git clone.

2. **Install dependencies**
   Open your terminal, navigate to the project folder, and run:
   `pip install -r requirements.txt`

3. **Set up your API Key**
   Copy the example env file and fill in your Vocareum API key:
   ```
   cp .env.example .env
   ```
   Then edit `.env` and replace `your_api_key_here` with the key provided by the instructor.
   The project uses the Vocareum proxy (`https://genai.vocareum.com/v1`) and the `@azure-1/gpt-4o` model.

4. **Add the Source Documents**
   Ensure the 6 required UMSI PDF handbooks and requirement guides are placed inside the `data/raw_documents/` folder.

## Usage

To test the document loading, chunking, and retrieval pipeline, run the main script:
`python -m streamlit run src/pipeline.py`
