import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

load_dotenv()  # Load environment variables

# Load chat history from MongoDB
chat_message_history = MongoDBChatMessageHistory(
    session_id="user00000",
    connection_string=os.environ["MONGODB_ACCESS_KEY"],
    database_name="0_chatbot_streamlit",
    collection_name="chat_history",
)


def get_response(model, query, chat_history, language):
    system_prompt = SystemMessagePromptTemplate.from_template(
        template="""
    You are a multilingual AI assistant. Answer the following question from user in {language}, based on the chat history.
    ---
    Chat history: {chat_history}
    ---
    Question: {query}
    """,
        input_variables=["language", "chat_history", "query"],
    )
    prompt = ChatPromptTemplate.from_messages([system_prompt])
    parser = StrOutputParser()

    chain = prompt | model | parser
    return chain.stream(
        {"chat_history": chat_history, "query": query, "language": language}
    )


# ! === PAGE CONFIGS ===
st.set_page_config(page_title="Chatbot with Streamlit", page_icon=":streamlit:")
st.title(body="Chatbot with Streamlit :streamlit:")

# ! === SIDEBAR ===
st.sidebar.title("Model Parameters")
language = st.sidebar.selectbox(
    label="Language",
    options=[
        "English",
        "French",
        "German",
        "Italian",
        "Spanish",
    ],
)

# ! === INITIALISATION ===
MAX_TURNS = 10  # Limit chat history to display (to reduce token usage)
if "chat_history" not in st.session_state:
    loaded_history = chat_message_history.messages  # Load messages from MongoDB
    if loaded_history:
        st.session_state.chat_history = loaded_history[-MAX_TURNS:]
    else:
        initial_message = "Hello! how can I help you?"
        st.session_state.chat_history = [AIMessage(content=initial_message)]
        chat_message_history.add_ai_message(AIMessage(content=initial_message))

if "model" not in st.session_state:
    st.session_state.model = OpenAI(name="gpt-4o-mini")

model = st.session_state.model

# ! === DISPLAY (MAIN) ===
# Iterate all the chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, HumanMessage):
        role = "user"

    # Display role + message
    with st.chat_message(role):
        st.markdown(message.content)

# ! === USER INPUT + AI RESPONSE ===
# User input
if prompt := st.chat_input("Type a message..."):
    user_message = HumanMessage(prompt)
    st.session_state.chat_history.append(user_message)
    chat_message_history.add_user_message(user_message)
    # Display user input immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(
            get_response(
                query=prompt,
                chat_history=st.session_state.chat_history,
                language=language,
                model=model,
            )
        )
    ai_message = AIMessage(response)
    st.session_state.chat_history.append(ai_message)
    chat_message_history.add_ai_message(ai_message)
