from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage,ToolMessage,FunctionMessage,ChatMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    '''updates the document with the provided content.'''
    global document_content
    document_content = content
    return F"Document updated. The current content is {document_content}"

@tool
def save(filename: str) -> str:
    '''saves the current content to a text file and finish the process.
    Args:
        filename (str): The name of the text file to save the content to.'''
    
    global document_content

    if not filename.endswith(".txt"):
        raise ValueError("Filename must end with .txt")
    with open(filename, "w") as f:
        f.write(document_content)
    return f"Document saved to {filename}"

tools=[update, save]

model=ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
     """Determine if we should continue the conversation """
     messages = state["messages"]
     if not messages:
       return "continue"
     
     for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
        "saved" in message.content.lower() and
        "document" in message.content.lower()):
          return "end"
     
     return "continue"
          

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

  
graph=StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_edge("agent", "tools")

graph.add_conditional_edges("tools", should_continue, {"continue": "agent", "end": END})

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
     run_document_agent()