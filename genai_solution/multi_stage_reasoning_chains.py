# Databricks notebook source
# MAGIC %pip install --upgrade --quiet langchain-core databricks-vectorsearch langchain-community youtube_search wikipedia
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

#Prompt
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me about a {genre} movie which {actor} is one of the actors.")
prompt_template.format(genre="romance", actor="Brad Pitt")

# COMMAND ----------

#LLM
from langchain_community.chat_models import ChatDatabricks

# play with max_tokens to define the length of the response
llm_dbrx = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500)

for chunk in llm_dbrx.stream("Who is Brad Pitt?"):
    print(chunk.content, end="\n", flush=True)

# COMMAND ----------

#Retriever
from langchain_community.retrievers import WikipediaRetriever
retriever = WikipediaRetriever()
docs = retriever.invoke(input="Brad Pitt")
print(docs[0])

# COMMAND ----------

#tools
from langchain_community.tools import YouTubeSearchTool
tool = YouTubeSearchTool()
tool.run("Brad Pitt movie trailer")
print(tool.description)
print(tool.args)

# COMMAND ----------

#Chaining
from langchain_core.output_parsers import StrOutputParser

chain = prompt_template | llm_dbrx | StrOutputParser()
print(chain.invoke({"genre":"romance", "actor":"Brad Pitt"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build a Multi-stage Chain

# COMMAND ----------

vs_endpoint_prefix = "vs_endpoint_"
vs_endpoint_fallback = "vs_endpoint_fallback"
vs_endpoint_name = "vs_endpoint_1"
vs_index_table_fullname =  "dbdemos.rag_carlosfuentes.pdf_text_self_managed_vs_index"
source_table_fullname = "dbdemos.rag_carlosfuentes.pdf_text_embeddings"

# COMMAND ----------

#First chain
from langchain_community.chat_models import ChatDatabricks
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import YouTubeSearchTool
from databricks.vector_search.client import VectorSearchClient
from langchain.schema.runnable import RunnablePassthrough

llm_dbrx = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 1000)
tool_yt = YouTubeSearchTool()

prompt_template_1 = PromptTemplate.from_template(
    """You are a Climate Change expert. You will get questions about Climate Change. Try to give simple answers and be professional.

    Question: {question}

    Answer:
    """
)

chain1 = ({"question": RunnablePassthrough()} | prompt_template_1 | llm_dbrx | StrOutputParser())
print(chain1.invoke({"question":"How climate change is affecting people?"}))

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from dbdemos.rag_carlosfuentes.pdf_text_embeddings

# COMMAND ----------

#Second chain
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
vsc = VectorSearchClient()
vs_index = vsc.get_index(endpoint_name=vs_endpoint_name,index_name=vs_index_table_fullname)
query = "How climate change is afeccting agriculture"

dvs_delta_sync = DatabricksVectorSearch(vs_index, text_column="content", embedding=embedding_model)
docs = dvs_delta_sync.similarity_search(query)

videos = tool_yt.run(docs[0].page_content.replace(',', ''))

prompt_template_2 = PromptTemplate.from_template(
    """You will get a list of videos related to the user's question. Encourage the user to watch the videos. List videos with their YouTube links.

    List of videos: {videos}
    """
)
chain2 = ({"videos": RunnablePassthrough()} | prompt_template_2 |  llm_dbrx | StrOutputParser())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chaining Chains ⛓️
# MAGIC
# MAGIC So far we create chains for each stage. To build a multi-stage system, we need to link these chains together and build a multi-chain system.

# COMMAND ----------

from langchain.schema.runnable import RunnablePassthrough
from operator import itemgetter

multi_chain = ({
    "c":chain1,
    "d": chain2
}| RunnablePassthrough.assign(d=chain2))

multi_chain.invoke({"question":"How climate change is impacting agriculture", "videos":videos})
