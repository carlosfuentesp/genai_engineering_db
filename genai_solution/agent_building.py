# Databricks notebook source
# MAGIC %pip install --upgrade --quiet langchain==0.1.16 langchain-core langchain_community==0.0.36 langchain-experimental youtube_search wikipedia==1.4.0 duckduckgo-search
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

pip install mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agent 1

# COMMAND ----------

#Brain

from langchain_community.chat_models import ChatDatabricks

# play with max_tokens to define the length of the response
llm_dbrx = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500)

# COMMAND ----------

#Tools

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.tools import YouTubeSearchTool

from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL

from langchain_community.tools import DuckDuckGoSearchRun

# Wiki tool for info retrieval
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool_wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# tool to search youtube videos
tool_youtube = YouTubeSearchTool()

# web search tool
search = DuckDuckGoSearchRun()

# tool to write python code
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

# toolset
tools = [tool_wiki, tool_youtube, search, repl_tool]

# COMMAND ----------

#Planning

from langchain.prompts import PromptTemplate

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt= PromptTemplate.from_template(template)

# COMMAND ----------

#Create the Agent

from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent

agent = create_react_agent(llm_dbrx, tools, prompt)
if not agent:
    raise ValueError("Agent creation failed. Check the configuration of the language model and tools.")


brixo  = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
brixo.invoke({"input": 
    """What would be a comedy movie to watch with my wife. Follow these steps.
    
    First, decide which movie you would recommend.

    Second, show me the trailler video of the movie that you suggest. 

    Next, collect data about the movie using search tool and  draw a bar chart using Python libraries. If you can't find latest data use some dummy data as we to show your abilities to the learners. Don't use ``` for python code. Input should be sanitized by removing any leading or trailing backticks. if the input starts with ”python”, remove that word as well. The output must be the result of executed code.

    Finally, tell a funny joke about agents.
    """})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agent 2

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install pandas tabulate langchain_openai

# COMMAND ----------

import pandas as pd

df = pd.read_csv("/dbfs/FileStore/pdfs/Crime_Data_from_2020_to_Present.csv")

# COMMAND ----------

#Brain
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatDatabricks

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct", max_tokens = 500)

prefix = """ Input should be sanitized by removing any leading or trailing backticks. if the input starts with ”python”, remove that word as well. Use the dataset provided. The output must start with a new line."""

dataqio = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    max_iterations=3,
    prefix=prefix,
    agent_executor_kwargs={
        "handle_parsing_errors": True
    }
)

# COMMAND ----------

dataqio.invoke("How many crimes are there per location?")

# COMMAND ----------

dataqio.invoke("What is the total number of rows?")
