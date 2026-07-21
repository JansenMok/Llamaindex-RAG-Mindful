# Llamaindex RAG Mindful

RAG chatbot that answers questions on Mindful Substack articles, grounded in the actual article text instead of the model's own knowledge.

## Setup
1. `$ git clone git@github.com:JansenMok/Llamaindex-RAG-Mindful.git`
2. `$ pip install llama-index llama-index-llms-google-genai llama-index-embeddings-google-genai streamlit`
3. Set `GOOGLE_API_KEY` env var
4. Add article `.txt` files to `articles/`
5. `$ streamlit run main.py`

## Pipeline
1. **Chunking**: articles loaded via `SimpleDirectoryReader`, split into ~500-token chunks
2. **Embedding**: chunks embedded using Google's `text-embedding-004`
3. **Retrieval**: vector store index (`GPTVectorStoreIndex`), default top-k similarity search per query
4. **Generation**: retrieved chunks + question passed to `gemini-2.0-flash` for a grounded answer

## Model Choice
`gemini-2.0-flash` for generation and `text-embedding-004` for embeddings, both via the `google-genai` SDK, chosen for fast inference and a generous free tier that lets you iterate on retrieval quality without API cost getting in the way.
## Interface
Streamlit web app: type a question, get a grounded answer pulled from the Mindful Articles corpus.

## Example
**Query:** "what does the author say about morning routines?"
**Retrieved:** top relevant chunks from that article
**Response:** grounded answer citing the article directly
