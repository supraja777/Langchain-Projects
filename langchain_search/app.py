from dotenv import load_dotenv
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI

import streamlit as st
from langchain.agents import load_tools, initialize_agent
import os

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))
serpapi_api_key = os.getenv("SERP_API_KEY")

st.title("Langchain Search")

search_query = st.text_input("Search Query")

if st.button("Search"):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    tools = load_tools(["serpapi"], llm, serpapi_api_key = serpapi_api_key)
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose = True)

    result = agent.run(search_query)
    st.write(result)


