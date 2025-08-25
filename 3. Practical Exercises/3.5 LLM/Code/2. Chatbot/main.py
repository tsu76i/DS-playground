import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)

load_dotenv()  # Load environment variables


def get_response(model, query, chat_history, language):
    system_prompt = SystemMessagePromptTemplate.from_template(
        template="""
    You are a multilingual AI assistant. Answer the following question from user in {language}, based on the chat history.
    ---
    Chat history: {chat_history}
    ---
    Question: {query}
    """,
        input_variables=["language"],
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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! how can I help you?")]

model = OpenAI(name="gpt-4o-mini")

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
    st.session_state.chat_history.append(HumanMessage(prompt))
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
    st.session_state.chat_history.append(AIMessage(response))
