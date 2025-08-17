from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from todoist_api_python.api import TodoistAPI
import os

load_dotenv()
todoist_api_token = os.getenv("TODO_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

@tool
def add_task(task:str,desc:str = None):
    """Add a task to Todoist."""

    todo = TodoistAPI(todoist_api_token)
    todo.add_task(content=task,description=desc)
    print("task:",task)
    print("Task added successfully.")
@tool
def show_tasks():
    """Show all tasks in Todoist."""
    todo = TodoistAPI(todoist_api_token)
    results = todo.get_tasks()
    tasks = []
    for task_list in results:
        for task in task_list:
            tasks.append(task.content)
    return tasks

tool = [add_task,show_tasks]

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=gemini_api_key,
    temperature=0.3)

system_prompt = """You are a helpful AI assistant. 
You will help the user to add tasks.
You will help the user to show existing tasks.
if the user asks to show the tasks, you will print the tasks in a bullet point format."""
#user_input = input("You: ")
prompt  = ChatPromptTemplate([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("user","{input}"),
    MessagesPlaceholder("agent_scratchpad")
])
def main():
    agent = create_openai_tools_agent(
        llm=llm,
        tools=tool,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=False)
    history = []
    while True:
        user_input = input("You: ")
        response = agent_executor.invoke({
        "input": user_input,
        "history": history})
        print("AI:", response['output'])
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response['output']))
    


if __name__ == "__main__":
    main()
