import getpass
import os


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY") or getpass.getpass("Enter your LANGSMITH_API_KEY: ")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY") or getpass.getpass("Enter your GEMINI_API_KEY: ")

os.environ["REDDIT_CLIENT_ID"] = os.getenv("REDDIT_CLIENT_ID") or getpass.getpass("Enter your REDDIT_ID: ")
os.environ["REDDIT_CLIENT_SECRET"] = os.getenv("REDDIT_CLIENT_SECRET") or getpass.getpass("Enter your REDDIT_CLIENT_SECRET: ")
os.environ["REDDIT_USER_AGENT"] = "john"


import requests

from typing import Optional, Type, Annotated
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper


class RedditInputs(BaseModel):
    """Inputs to the reddit tool."""

    query: str = Field(
        description="should be query string that post title should \
        contain, or '*' if anything is allowed."
    )
    sort: str = Field(
        description='should be sort method, which is one of: "relevance" \
        , "hot", "top", "new", or "comments".'
    )
    time_filter: str = Field(
        description='should be time period to filter by, which is \
        one of "all", "day", "hour", "month", "week", or "year"'
    )
    subreddit: str = Field(
        description='the subreddit to search for, try r/tonightsdinner'
    )
    limit: str = Field(
        description="a positive integer indicating the maximum number \
        of results to return"
    )


class RedditFoodSearchRun(BaseTool):  # type: ignore[override, override]
    """Tool that queries for food ideas on a subreddit."""

    name: str = "reddit_search"
    description: str = (
        "A tool that searches for food ideas on Reddit."
        "Useful only when you need to get food ideas from on a subreddit."
    )
    api_wrapper: RedditSearchAPIWrapper = Field(default_factory=RedditSearchAPIWrapper)  # type: ignore[arg-type]
    args_schema: Type[BaseModel] = RedditInputs

    def _run(
        self,
        query: str,
        sort: str,
        time_filter: str,
        limit: str,
        subreddit: Annotated[str, InjectedToolArg] = "r/tonightsdinner",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(
            query=query,
            sort=sort,
            time_filter=time_filter,
            subreddit=subreddit,
            limit=int(limit),
        )

class RecipeInput(BaseModel):
    meal_name: str = Field(description="Name of the meal to search for, should only be 3 words max.")

class RecipeSearchRun(BaseTool):
    name: str = "recipe_search"
    description: str = "A tool that searches for recipes online. Returns instructions (strInstructions), ingredient and cusines (strArea) of the meal."
    args_schema: Type[BaseModel] = RecipeInput
    return_direct: bool = True

    def _run(
        self, meal_name: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Search for a recipe by the meal or food name."""

        print(meal_name)
        url = f"https://www.themealdb.com/api/json/v1/1/search.php?s={meal_name}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data["meals"]:
                meal = data["meals"][3]  # Get the first meal from the list
                return meal
            else:
                return "No meal found."
        else:
            return "Error fetching data from API."



import vertexai

from typing_extensions import TypedDict

from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool


vertexai.init(project="dinner-000")


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

api_wrapper = RedditSearchAPIWrapper()
reddit_search = RedditSearchRun(
    name="reddit_search",
    description="A tool that searches food and meal ideas only, not used for searching for recipes.",
    api_wrapper=api_wrapper,
)

food_search = RedditFoodSearchRun()
recipe_search = RecipeSearchRun()

tools = [food_search, recipe_search]
llm = ChatVertexAI(model="gemini-1.5-flash")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    # chain = llm_with_tools | StrOutputParser()
    # return {"messages": [chain.invoke(state["messages"])]}
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def build_graph():
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")

    return graph_builder.compile(checkpointer=MemorySaver())


import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# app config
st.set_page_config(page_title="Chefbot", page_icon="🍽️")
st.title("What's for dinner?")

config = {"configurable": {"thread_id": "24"}}

def get_response(graph, user_query):    
    return graph.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
        config,
        # stream_mode="values",
    )

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I'm a chefbot. How can I help you?"),
    ]
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(st.session_state.graph, user_query)
        for msg in reversed(response['messages']):
            if isinstance(msg, AIMessage):
                st.markdown(msg.content)
                st.session_state.chat_history.append(AIMessage(content=msg.content))
            if isinstance(msg, ToolMessage):
                break
                # st.markdown(msg.content)
