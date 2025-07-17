import os
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import OllamaLLM
import streamlit as st 
from langchain_core.prompts import ChatPromptTemplate    ## basically used to make my own chat prompt 
from langchain_core.output_parsers import StrOutputParser    ## it is for giving the output in string format 

## langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGCHAIN_PROJECT"]

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , "You are a helpfull assistant . Please respond to the question asked"),
        ("user" , "Question:{question}")
    ]
)

## Streamlit Framework
st.title("Langchain Demo With GEMMA3")
intput_text = st.text_input("What question you have in mind?")

## ollama gemma model
llm = OllamaLLM(model="gemma3:1b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if intput_text:
    st.write(chain.invoke({"question":intput_text}))