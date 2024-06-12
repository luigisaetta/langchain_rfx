"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-27
Python Version: 3.11
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#
# The prompt for the condensed query on the Vector Store
#
CONTEXT_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

CONTEXT_Q_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXT_Q_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

#
# The prompt for the answer from the LLM
# (12/06): I have added the 3rd line
#
QA_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
Provide a long and detailed answer with all the possible details. \
If you don't know the answer, just say that you don't know. \

{context}"""

QA_SYSTEM_PROMPT_RATING = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context. \
Provide only a rating of the quality of the provided context to nswer the question based on a scale from 1 to 10.\ 

{context}"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
