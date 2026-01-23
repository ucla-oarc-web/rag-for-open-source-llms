from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import Optional
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from shared.knowledge_base import retriever

router = APIRouter()

def print_thoughts(result):
    # Debug: print full message structure
    print("=== AGENT RESULT ===")
    tool_call_map = {}  # Map tool_call id to name
    for i, msg in enumerate(result["messages"]):
        print(f"Message {i}:")
        print(f"  type: {getattr(msg, 'type', 'unknown')}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"  tool_calls: {msg.tool_calls}")
            for tc in msg.tool_calls:
                tool_call_map[tc["id"]] = tc["name"]
        if getattr(msg, 'type', '') == 'tool':
            tool_name = tool_call_map.get(getattr(msg, 'tool_call_id', ''), 'unknown')
            print(f"  tool_name: {tool_name}")
        print(f"  content: {repr(msg.content)}")
    print("=== END RESULT ===")

def extract_tools_used(result):
    tools_used = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] not in tools_used:
                    tools_used.append(tc["name"])
    return tools_used

# Conversation History
# In-memory conversation (resets on container restart)
conversation_history: dict[str, list] = {}

def get_conversation_messages(session_id: Optional[str], question: str):
    if session_id:
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        return conversation_history[session_id] + [{"role": "user", "content": question}]
    return [{"role": "user", "content": question}]

def save_conversation(session_id: Optional[str], question: str, answer: str):
    if session_id:
        conversation_history[session_id].append({"role": "user", "content": question})
        conversation_history[session_id].append({"role": "assistant", "content": answer})

@tool
def knowledge_base_search(query: str) -> str:
    """Search the Mythical Man Month knowledge base for information about the book."""
    docs = retriever.invoke(query)
    return "\n\n".join(f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc in docs)

# DuckDuckGo web search tool
web_search = DuckDuckGoSearchRun()

ollama_llm = ChatOllama(
    base_url="http://127.0.0.1:11434",
    model="qwen3-coder:30b",
    temperature=0.0
)

tools = [knowledge_base_search, web_search]
agent = create_agent(
    model=ollama_llm,
    tools=tools,
    system_prompt="""
    INCLUDE
    You are a helpful assistant that answers questions using the provided tools.
    - You MUST use tools to answer questions. Do NOT use your own knowledge.
    - For questions about the Mythical Man Month, Brooks' Law, software project
      management from the book, or Fred Brooks: use knowledge_base_search
    - For current events, weather, or other general topics: use duckduckgo_search
    - If the tool returns no relevant information, respond exactly: I do not know.
    - Do not repeat the question in your answer.

    RESTRICT
    - Limit responses to 150 words or less.

    ADD
    Think about this step by step.

    REPEAT & POSITION
    Always use a tool first. Base your answer ONLY on the tool's response."""
)

class AgentQueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

@router.post("/agent")
def query_agent(request: AgentQueryRequest = Body(...)):
    """
    Ask a question using the agent that can search the knowledge base or the web.
    Optionally pass session_id to maintain conversation history.
    """

    # Load messages from conversation history if they exist
    messages = get_conversation_messages(request.session_id, request.question)

    # Calls the agent/llm
    result = agent.invoke({"messages": messages})

    # Helpful for debugging issues with the agent.
    print_thoughts(result)

    # Get the final message content
    answer = result["messages"][-1].content or "No response generated"

    # Save conversation history if session_id provided, only includes the last
    # message from the agent.
    save_conversation(request.session_id, request.question, answer)

    return {
        "answer": answer,
        "tools_used": extract_tools_used(result),
        "session_id": request.session_id
    }
