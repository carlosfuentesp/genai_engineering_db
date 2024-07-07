# Databricks notebook source
# MAGIC %pip install -U --quiet langchain==0.1.16 databricks-vectorsearch==0.22 pydantic==1.10.9 mlflow==2.12.1  databricks-sdk==0.28.0 "unstructured[pdf,docx]==0.10.30"
# MAGIC
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# components we created before
# assign vs search endpoint by username
vs_endpoint_prefix = "vs_endpoint_"
vs_endpoint_fallback = "vs_endpoint_fallback"
vs_endpoint_name = "vs_endpoint_1"

print(f"Vector Endpoint name: {vs_endpoint_name}.")

vs_index_fullname = "dbdemos.rag_carlosfuentes.pdf_text_self_managed_vs_index"

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings


# Test embedding Langchain model
#NOTE: your question embedding model must match the one used in the chunk in the previous model 
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
print(f"Test embeddings: {embedding_model.embed_query('What is Climate Change?')[:20]}...")

def get_retriever(persist_dir: str = None):
    #Get the vector search index
    vsc = VectorSearchClient()
    vs_index = vsc.get_index(
        endpoint_name=vs_endpoint_name,
        index_name=vs_index_fullname
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model
    )
    # k defines the top k documents to retrieve
    return vectorstore.as_retriever(search_kwargs={"k": 2})


# test our retriever
vectorstore = get_retriever()
similar_documents = vectorstore.invoke("How climate change impacts agriculture?")
print(f"Relevant documents: {similar_documents}")

# COMMAND ----------

from langchain.chat_models import ChatDatabricks


# Test Databricks Foundation LLM model
chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 300)
print(f"Test chat model: {chat_model.invoke('What is Climate Change?')}")

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks


TEMPLATE = """You are an assistant for Climate Change teaching class. You are answering questions related to Cliamte Change and how it impacts agriculture. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
Use the following pieces of context to answer the question at the end:

<context>
{context}
</context>

Question: {question}

Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

question = {"query": "How climate change impacts agriculture?"}
answer = chain.invoke(question)
print(answer)

# COMMAND ----------

type(chain)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare the Evaluation Dataset

# COMMAND ----------

eval_set = """question,ground_truth,evolution_type,episode_done
"How does an increase in mean seasonal temperature affect crop yields?","Increase in mean seasonal temperature can reduce the duration of many crops and hence reduce the yield. In areas where temperatures are already close to the physiological maxima for crops, warming will impact yields more immediately.",simple,TRUE
"What are the three ways in which the Greenhouse Effect is important for agriculture?","First, increased atmospheric CO2 concentrations can have a direct effect on the growth rate of crop plants and weeds. Secondly, CO2-induced changes of climate may alter levels of temperature, rainfall, and sunshine that can influence plant and animal productivity. Finally, rises in sea level may lead to loss of farmland by inundation and increasing salinity of groundwater in coastal areas.",simple,TRUE
"How does climate change impact food production globally?","Climate change is likely to directly impact food production across the globe. Increase in the mean seasonal temperature can reduce the duration of many crops and hence reduce the yield. Developing countries, many of which have average temperatures that are already near or above crop tolerance levels, are predicted to suffer an average 10 to 25% decline in agricultural productivity by the 2080s.",simple,TRUE
"What is the impact of higher CO2 concentrations on rice yields?","Higher CO2 concentrations have generally shown significant increases in rice biomass (25-40%) and yields (15-39%) at ambient temperature, but those increases tended to be offset when temperature was increased along with rising CO2. Yield losses caused by concurrent increases in CO2 and temperature are primarily caused by high-temperature-induced spikelet sterility.",reasoning,TRUE
"What are some adaptation strategies for farmers to cope with climate change?","1. Shifting planting dates, choosing varieties with different growth durations, or changing crop rotations. 2. An Early warning system to monitor changes in pest and disease outbreaks. 3. Participatory and formal plant breeding to develop climate-resilient crop varieties. 4. Developing short-duration crop varieties that can mature before the peak heat phase sets in.",simple,TRUE
"How is food security linked with climate change?","Food security is both directly and indirectly linked with climate change. Any alteration in climatic parameters such as temperature and humidity, which govern crop growth, will have a direct impact on the quantity of food produced. Indirect linkage pertains to catastrophic events such as floods and droughts, projected to multiply as a consequence of climate change, leading to huge crop loss and threatening food security.",simple,TRUE
"How does climate change affect agriculture in India?","India’s agriculture is highly dependent on monsoon. Any change in monsoon trends drastically affects agriculture. Increased temperatures affect Indian agriculture by reducing yields of wheat, soybean, mustard, groundnut, and potato by 3-7% for every 1°C rise. Rice production is projected to decrease significantly with temperature rises, and rain-fed crops are the most vulnerable.",simple,TRUE
"What are the projected global warming scenarios for temperature rise by 2100?","Projected scenarios of global warming indicate that the global average surface temperature could rise by 1.4 to 5.8°C by 2100. The projected rate of warming is unprecedented during the last 10,000 years.",simple,TRUE
"How does climate change impact the agricultural productivity of developing countries?","Developing countries, many of which have average temperatures that are already near or above crop tolerance levels, are predicted to suffer an average 10 to 25% decline in agricultural productivity by the 2080s. Countries like India could see a drop of 30 to 40% in agricultural productivity.",reasoning,TRUE
"What are some mitigation strategies for agriculture to combat climate change?","1. Assist farmers in coping with current climatic risks by providing value-added weather services. 2. Develop climate-resilient crop varieties through participatory and formal plant breeding. 3. Implement efficient water and fertilizer use practices. 4. Adopt resource conservation technologies such as no-tillage and crop diversification. 5. Provide greater coverage of weather-linked agriculture insurance.",simple,TRUE
"""

import pandas as pd
from io import StringIO

obj = StringIO(eval_set)
eval_df = pd.read_csv(obj)

# COMMAND ----------

display(eval_df)

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install datasets

# COMMAND ----------

from datasets import Dataset

test_questions = eval_df["question"].values.tolist()
test_groundtruths = eval_df["ground_truth"].values.tolist()

answers = []
contexts = []

# answer each question in the dataset
for question in test_questions:
    # save the answer generated
    chain_response = chain.invoke({"query" : question})
    answers.append(chain_response["result"])
    
    # save the contexts used
    vs_response = vectorstore.invoke(question)
    contexts.append(list(map(lambda doc: doc.page_content, vs_response)))

# construct the final dataset
response_dataset = Dataset.from_dict({
    "inputs" : test_questions,
    "answer" : answers,
    "context" : contexts,
    "ground_truth" : test_groundtruths
})

# COMMAND ----------

display(response_dataset.to_pandas())

# COMMAND ----------

#Calculate evaluation metrics
import mlflow
from mlflow.deployments import set_deployments_target

set_deployments_target("databricks")

dbrx_answer_similarity = mlflow.metrics.genai.answer_similarity(
    model="endpoints:/databricks-dbrx-instruct"
)

dbrx_relevance = mlflow.metrics.genai.relevance(
    model="endpoints:/databricks-dbrx-instruct"   
)

results = mlflow.evaluate(
        data=response_dataset.to_pandas(),
        targets="ground_truth",
        predictions="answer",
        extra_metrics=[dbrx_answer_similarity, dbrx_relevance],
        evaluators="default",
    )

# COMMAND ----------

display(results.tables['eval_results_table'])

# COMMAND ----------

#save the model to Model Registry in UC
from mlflow.models import infer_signature
import mlflow
import langchain


# set model registery to UC
mlflow.set_registry_uri("databricks-uc")
model_name = "dbdemos.rag_carlosfuentes.rag_app_demo"

with mlflow.start_run(run_name="rag_app_demo") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever, 
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )
