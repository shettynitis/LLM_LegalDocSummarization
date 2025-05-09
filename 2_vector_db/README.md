# RAG Pipeline Overview

This folder includes code that builds a Retrieval-Augmented Generation (RAG) system for summarizing Supreme Court “facts” data.  

- **Data**  
  – Source: “Supreme Court Judgment Prediction” CSV from Kaggle (≈10,000 cases; ~200 MB uncompressed)  
  – Fields: `name` (case identifier) and `facts` (text to summarize)  

- **Models**  
  – **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` to convert text chunks into 384-dim vectors  
  – **Generator**: Llama-2 (fine-tuned on Zenodo legal summaries) for producing concise outputs  

- **Techniques**  
  1. **Cleaning & Chunking**: normalize whitespace, remove noise, split long texts into overlapping 512-token segments  
  2. **Indexing**: embed each chunk and store in a FAISS L2 index for fast vector similarity search  
  3. **Retrieval**: at query time, embed the user prompt, fetch top-K relevant chunks from FAISS  
  4. **Prompt Construction**: assemble retrieved snippets with an instruction template  
  5. **Generation**: feed the prompt into fine-tuned Llama-2 to generate the final summary  

Each step is designed for modularity and reproducibility, enabling you to swap datasets, tweak chunk sizes, or plug in different embedding/generation models.

The resultant index file is saved in the model_rag folder.
