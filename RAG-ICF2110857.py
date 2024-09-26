# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:01:27 2024

@author: 55231
https://github.com/wsxqaza12/RAG_example/blob/master/RAG_example.ipynb
https://medium.com/@cch.chichieh/rag%E5%AF%A6%E4%BD%9C%E6%95%99%E5%AD%B8-langchain-llama2-%E5%89%B5%E9%80%A0%E4%BD%A0%E7%9A%84%E5%80%8B%E4%BA%BAllm-d6838febf8c4
"""

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

 
###### load files what improves LLMs for RAG   
loader = PyMuPDFLoader("Virtual_characters.pdf")
PDF_data = loader.load()

###### split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
all_splits = text_splitter.split_documents(PDF_data)

###### Embed and store the texts
###### Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_directory)

##### load LLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
    model_path="llama-2_q4.gguf",
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

# Insert default prompts
from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.prompts import PromptTemplate

DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> 
    You are a helpful assistant eager to assist with providing better Google search results.
    <</SYS>> 
    
    [INST] Provide an answer to the following question in 150 words. Ensure that the answer is informative, \
            relevant, and concise:
            {question} 
    [/INST]""",
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a helpful assistant eager to assist with providing better Google search results. \
        Provide an answer to the following question in about 150 words. Ensure that the answer is informative, \
        relevant, and concise: \
        {question}""",
)

QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=DEFAULT_SEARCH_PROMPT,
    conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
)

prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)
prompt


# link prompts to LLM
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What is Taiwan known for?"
llm_chain.invoke({"question": question})


# LLM + RAG vector database
retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

# input prompt for LLM+RAG
query = "Tell me about Alison Hawk's career and age"
qa.invoke(query)
