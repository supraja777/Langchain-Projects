from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

from langchain.chains.summarize import load_summarize_chain
from langchain_community.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter

import os

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

st.subheader("Summarize text")

source_text = st.text_area("Source text", label_visibility="collapsed", height = 250)

if st.button("Summarize"):
        with st.spinner("Please wait.........."):
            
            text_splitter = CharacterTextSplitter()
            texts = text_splitter.split_text(source_text)

            docs = [Document(page_content=t) for t in texts[:3]]

            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            chain = load_summarize_chain(llm, chain_type = "map_reduce")
            summary = chain.run(docs)

            st.success(summary)




