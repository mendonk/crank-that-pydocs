from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from bs4 import BeautifulSoup

llm = GPT4All(
    model="/Users/mendon.kissling/Library/Application Support/nomic.ai/GPT4All/orca-mini-3b.ggmlv3.q4_0.bin",
    max_tokens=2048,
)

question = "What are the approaches to Task Decomposition?"

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
llm_chain.run(question)