from dotenv import load_dotenv
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI

import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

st.subheader("Summarize URL")

url = st.text_input("URL", label_visibility="collapsed")

if st.button("Summarize"):
    if not validators.url(url):
        st.error("Please enter a valid URL")
    else:
        with st.spinner("Please wait.........."):
            if "youtube.com" in url:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info = True)
            else:
                loader = UnstructuredURLLoader(urls = [url], headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})

            data = loader.load()

            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

            prompt_template = """Write a summary of the following in 250 - 300 words: 
            
            {text}
            
            """

            prompt = PromptTemplate(template = prompt_template, input_variables = ["text"])
            chain = load_summarize_chain(llm, chain_type = "stuff", prompt = prompt)

            summary = chain.run(data)

            st.success(summary)




