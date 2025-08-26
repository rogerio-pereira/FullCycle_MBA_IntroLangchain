from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers with short joke when possible."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0.9)

def prepare_inputs(payload: dict) -> dict:
    raw_history = payload.get("raw_history", [])    # raw_history is defined in conversational_chain
    # trimmed = trim_messages(raw_history,max_tokens=1000)
    trimmed = trim_messages(
        raw_history,
        token_counter=len,      # Messages count
        max_tokens=2,           # max 2 messages, this is the sliding window
        strategy="last",
        start_on="human",
        include_system=True,    # Make sure to include initial prompt: You are a helpful assistant...
        allow_partial=False,
    )
    return{"input": payload.get("input", ""), "history": trimmed}

prepare = RunnableLambda(prepare_inputs)

chain = prepare | prompt | llm

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id:str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

conversational_chain = RunnableWithMessageHistory(
    chain, 
    get_session_history,
    input_messages_key="input",
    history_messages_key="raw_history"  # Will be used in prepare_inputs
)

config = {"configurable": {"session_id": "demo-session"}}

# Interactions
resp1 = conversational_chain.invoke({"input": "Hello, my name is Rogerio. Reply with ok and do not mention my name."}, config=config)
print("Assistant: ", resp1.content)

resp2 = conversational_chain.invoke({"input": "Tell me an one-sentence fun fact. Do not mention my name."}, config=config)
print("Assistant: ", resp2.content)

# In theory it won't know your name because it's using a sliding window of 2 messages, and name was lost, because message 1 is not in the window
resp3 = conversational_chain.invoke({"input": "What is my name?"}, config=config)
print("Assistant: ", resp3.content)
