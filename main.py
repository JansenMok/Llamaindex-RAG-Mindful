import os
import streamlit as st

# SimpleDirectoryReader: reads .txt article
# GPTVectorStoreIndex: creates index to store/receive text chunks
from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# global settings
# Settings.llm = Gemini(model="models/gemini-2.0-flash")
Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")

articles = SimpleDirectoryReader("articles").load_data()  # load txt
index = GPTVectorStoreIndex.from_documents(articles)  # txt -> chunks (500 tokens)

query_engine = index.as_query_engine()  # RAG pipeline

# print(query_engine.query("Give me a summary of article1."))

# cli frontend
# while (True):
#     prompt = input("💬: ")
#     if (prompt == "end"):
#         exit()
#     print(query_engine.query(prompt))

# streamlit frontend
st.title("Amitabha Mindful QnA Chatbot")
query = st.text_input("💬: ")
if query:
    answer = query_engine.query(query)
    st.markdown(f"**Answer:** {answer}")
