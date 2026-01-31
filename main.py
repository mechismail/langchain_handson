import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from dotenv import load_dotenv
load_dotenv()  

# 0.Import libraries
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools.tavily_search import TavilySearchResults

print("All LangChain imports working ")

#1. API keys are stored in the .env file
print("API keys are stored in the .env file ")

#2.LLM Model 

from langchain.llms import HuggingFacePipeline
from transformers import pipeline

hf_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base", max_new_tokens=256)

llm = HuggingFacePipeline(pipeline=hf_pipeline)
print("HuggingFacePipeline LLM working ")

# 3. PromptTemplate + LLMChain (basic)
prompt = PromptTemplate.from_template("Answer clearly: {question}")
qa_chain = LLMChain(llm=llm, prompt=prompt)
print("LLM and PromptTemplate working ")

# 4. Load sample documents for RAG
loader = TextLoader("sample.txt") 
documents = loader.load()
print("Documents loaded ")

# 5. Embed and store in FAISS vector DB
embedding = HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(documents, embedding)
retriever = vectordb.as_retriever()
print("FAISS vector DB created ")

# 6. RetrievalQA chain (RAG)
rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
print("RetrievalQA chain created ")

# 7. External Tool - Web Search (Tavily)
search_tool = Tool( name="Tavily Search", func=TavilySearchResults(max_results=3).run,
                    description="Search the internet for additional information")
print("Tavily search tool created ")

# 8. Wrap RAG and QA as tools
tools = [
    Tool(name="Simple QA", func=qa_chain.run,description="Answer with basic LLMChain" ),
    Tool( name="RAG Search",  func=rag_chain.run,description="Answer using document retrieval"), search_tool]

# 9. Agent that chooses best tool (corrected)
agent = initialize_agent( tools=tools,  llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
print("Agent initialized ")

# 10. Run it (stable smart routing)
query = "What is LangGraph in LangChain?"
print("\nUser Question:", query)

if any(word in query.lower() for word in ["langgraph", "document", "langchain"]):
    answer = rag_chain.run(query)
else:
    answer = qa_chain.run(query)
print("\nFinal Answer:", answer)