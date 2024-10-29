import os
from dotenv import load_dotenv
from RAGAlchamy.ragalchemy.extractors.pptx import PPTExtractor
from langchain.text_splitter import MarkdownHeaderTextSplitter
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load NLP and embedding models
nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Function to parse PowerPoint and extract content
def parse_ppt(file_path):
    extractor = PPTExtractor(file_path)
    extractor.extract()
    return extractor.combine()

# Function to chunk text
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

# Function to embed and store chunks using FAISS
def embed_and_store(chunks):
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Function to retrieve and generate response
def retrieve_and_generate(query, index, embeddings, chunks):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=3)
    context = "\n\n".join([chunks[i] for i in indices[0]])
    response = openai.Completion.create(
        engine="gpt-4o-mini",
        prompt=f"Context: {context}\n\nQuestion: {query}\nAnswer:",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Main function to run the pipeline
def main():
    ppt_content = parse_ppt("path to pptx")
    chunks = chunk_text(ppt_content)
    metadata = generate_metadata(chunks)
    index, embeddings = embed_and_store(chunks)
    response = retrieve_and_generate("What is the sales for 4th Qtr?", index, embeddings, chunks)
    print(response)

if __name__ == "__main__":
    main()