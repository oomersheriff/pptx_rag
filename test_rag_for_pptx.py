import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from langchain.text_splitter import MarkdownHeaderTextSplitter
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import openai

# Load environment variables
load_dotenv()
endpoint = os.getenv("TEXT_ANALYTICS_ENDPOINT")
key = os.getenv("TEXT_ANALYTICS_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize FAISS index
dimension = 768
index = faiss.IndexFlatL2(dimension)

# Helper function to authenticate Azure Document Intelligence client
def authenticate_document_intelligence_client():
    if not key:
        raise ValueError("AZURE_DOCUMENT_INTELLIGENCE_KEY environment variable is not set or is empty.")
    return DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

# Initialize Azure Document Intelligence client
client = authenticate_document_intelligence_client()

# Load a pre-trained NLP model for entity recognition
nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Load a pre-trained BERT model for embeddings
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Function to parse PowerPoint and extract content
def parse_ppt(file_path):
    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            document=f.read(),
            content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
    result = poller.result()
    return result.content

# Function to chunk text using LangChain
def chunk_text(content):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    splits = text_splitter.split_text(content)
    return splits

# Function to generate metadata
def generate_metadata(chunks):
    metadata = []
    for chunk in chunks:
        entities = nlp(chunk)
        metadata.append({"chunk": chunk, "entities": entities})
    return metadata

# Function to embed and store chunks
def embed_and_store(chunks):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)

# Function to retrieve and generate response
def retrieve_and_generate(query):
    query_embedding = model.encode([query])[0].astype('float32')
    distances, indices = index.search(np.array([query_embedding]), k=3)
    context = "\n\n".join([chunks[i] for i in indices[0]])
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Context: {context}\n\nQuestion: {query}\nAnswer:",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Main function to run the pipeline
def main():
    ppt_content = parse_ppt("path to pptx file")
    chunks = chunk_text(ppt_content)
    metadata = generate_metadata(chunks)
    embed_and_store(chunks)
    response = retrieve_and_generate("What is the sales for 4th Qtr?")
    print(response)

if __name__ == "__main__":
    main()