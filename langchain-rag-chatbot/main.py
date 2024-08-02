
from google.colab import userdata
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ReduceDocumentsChain, StuffDocumentsChain, MapReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

open_ai_key = userdata.get('open_ai_key')

# Define prompt
prompt_template = """Write a concise summary in a maximum of 3 bullets of the following text enclosed within three backticks:
```{text}```
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

# Define LLM & LLM chain
llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo-preview", api_key=open_ai_key)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Load PDF.
loader = PyPDFLoader("12224.full.pdf")
pages = loader.load_and_split()

# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
res = stuff_chain.invoke(pages[0:3])

print(res["output_text"])

# Define MapReduceDocumentsChain
reduce_chain = ReduceDocumentsChain(
    combine_documents_chain=stuff_chain
)
reduce_documents_chain = MapReduceDocumentsChain(
    llm_chain=llm_chain,
    reduce_documents_chain=reduce_chain
)
res = reduce_chain.invoke(pages[0:3])

print(res["output_text"])

# Extra Credit - Barack Obama Chat Bot

import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load web content.
loader = WebBaseLoader(
    web_paths=("https://barackobama.medium.com/my-statement-on-president-bidens-announcement-1eb78b3ba3fc",),
)

docs = loader.load()
print(docs)

# Create retriever.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=open_ai_key))
retriever = vectorstore.as_retriever()

# System prompt for maintaining chat histories.
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Define prompt and instantiate chains.
system_prompt = (
    "You are former 44th President Barack Obama, engaging in "
    "casual conversation about a Medium article you wrote "
    "surrounding President Biden dropping out of the 2024 "
    "presidential race. Emulate his mannerisms, make sure "
    "responses are concise, and appropriately facilitate a "
    "back-and-forth. Be concise and keep responses to 2-3 "
    "sentences. "
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Prepare chat history.
chat_history = []

question = "Hey Obama. Why do you think Joe Biden stepped down?";
res = rag_chain.invoke({"input": question, "chat_history": chat_history})
answer = res["answer"]
chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=answer),
    ]
)

print(answer)

question = "What are some good things he's done?";
res = rag_chain.invoke({"input": question, "chat_history": chat_history})
answer = res["answer"]
chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=answer),
    ]
)

print(answer)

question = "How long have you known him?";
res = rag_chain.invoke({"input": question, "chat_history": chat_history})
answer = res["answer"]
chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=answer),
    ]
)

print(answer)