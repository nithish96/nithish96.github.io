# Chapter 6: Knowledge and Memory 

In most cases, you will want your agents to both remember what has happened and access additional information beyond what is stored in the model’s weights. Knowledge and memory are two distinct ways to enrich an agent’s context.

Knowledge brings in factual or domain-specific content, such as technical specifications and policy documents. Memory, on the other hand, captures the agent’s own history, including prior user interactions, tool outputs, and state updates.

In this chapter, we will cover the fundamentals of memory for agentic systems, ranging from simple rolling context windows to semantic memory, retrieval-augmented generation, and advanced knowledge graph approaches.

## Foundational Approaches to Memory 

Simple approach is to rely on a rolling context window for the foundation model and keyword based memory. Despite the simplicity they are more than sufficient for wide range of use cases. 

### Managing Context Windows 

The context window refers to the information that is passed to a foundation model as input in a single call. The maximum number of tokens a model can ingest and attend to in a single call is called the context length. On average, one token corresponds to about three-quarters of a word.

The context window is a critical resource that developers must use effectively. In the simplest approach, the context window contains the current question along with all previous interactions in the current session. When the window becomes full, only the most recent interactions are retained.

With large prompts or verbose responses, running out of context space can happen very quickly. Highlighting the most relevant context and placing it toward the end of the prompt can increase the likelihood that it will be attended to by the model.

Below is a simple example using LangGraph.


``` python 

from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START

llm = ChatOpenAI(model="gpt-5")

def call_model(state: MessagesState):
   response = llm.invoke(state["messages"])
   return {"messages": response}

# Fails to maintain state across the conversation
input_message = {"type": "user", "content": "hi! I'm bob"}
for chunk in graph.stream({"messages": [input_message]}, stream_mode="values"):
   chunk["messages"][-1].pretty_print()

input_message = {"type": "user", "content": "what's my name?"}
for chunk in graph.stream({"messages": [input_message]}, stream_mode="values"):
   chunk["messages"][-1].pretty_print()


```

### Traditional Full-Text Search 

Traditional full text search forms the backbone of many large scale retrieval systems. Inverted index which preprocess all text via tokenization, normalization and stop word removal. THen each term is mapped with list of documents in which the term appears. This structure enables lightning fast lookups. 


To rank these results most system employs BM25 scoring function. Bm25 weighs each passage by uts term frequency, inverse document frequency and document length normalization. When a user query arrives, it is analyzed with the same text pipeline used for indexing, and BM25 produces a sorted list of the top K candidate passages. 

These hits are injected directly into foundation model prompt ensuring the model ses the most pertinent historiacal context withoug exhausting context length 

``` python 


from rank_bm25 import BM25Okapi
from typing import List

corpus: List[List[str]] = [
    "Agent J is the fresh recruit with attitude".split(),
    "Agent K has years of MIB experience and a cool neuralyzer".split(),
    "The galaxy is saved by two Agents in black suits".split(),
]
# 2. Build the BM25 index
bm25 = BM25Okapi(corpus)

# 3. Perform retrieval for a fun query
query = "Who is a recruit?".split()
top_n = bm25.get_top_n(query, corpus, n=2)

print("Query:", " ".join(query))
print("Top matching lines:")
for line in top_n:
    print(" •", " ".join(line))

``` 

While this keyword-driven approach excels at pinpointing exact or highly specific terms, it can miss broader themes, paraphrases, or conceptual links that weren’t expressed in the original text. 

## Semantic Memory and Vector Stores

Semantic memory is a type of memory that involves the storage and retrieval of general knowledge and past experiences. Leading way to do this is by reading vector databases which enable rapid indexing and retrieval at large scale. 

### Introduction to Semantic Search 

Semantic search aims to understand the context and intent behind a query, leading to more accurate and meaningful results. It focuses on the meaning of words and phrases rather than exact keyword matches.

The foundation of these approaches is embeddings, which capture word meanings based on their usage across large text corpora. Large language models have further improved the performance of embedding models across a wide range of text types by increasing both the size of the models and the quantity and variety of training data.

Semantic search has proven to be an invaluable technique for improving memory performance, particularly when retrieving semantically relevant information from documents that do not share overlapping keywords.


### Implementing Semantic memory with Vector Store 

We begin by generating semantic embeddings for the concepts and knowledge to be stored. These embeddings are produced by foundation models or other NLP techniques that encode text into dense vector representations.

Vector stores such as VectorDB, FAISS, or ANNOY are optimized for storing and searching high-dimensional vectors. These stores are designed for fast similarity searches, enabling the retrieval of embeddings that are semantically similar to a given query.

When an agent receives a query, it can use the vector store to perform a similarity search based on the query’s embedding. By identifying the most relevant embeddings in the vector store, the agent can access stored semantic memory and generate contextually appropriate responses.

This can be implemented as follows:

``` python

from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from vectordb import Memory

llm = ChatOpenAI(model="gpt-5")
def call_model(state: MessagesState):
   response = llm.invoke(state["messages"])
   return {"messages": response}

memory = Memory(chunking_strategy={'mode':'sliding_window', 'window_size': 128, 
'overlap': 16})
text = """Here is a simple text"""

metadata = {"title": "Introduction to Machine Learning", "url": ""}
memory.save(text, metadata)

text2 = """Here is a sample text"""
metadata2 = {"title": "Artificial Intelligence ", "url": ""}
memory.save(text2, metadata2)
query = "What is the relationship between AI and machine learning?"
results = memory.search(query, top_n=3)
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile()
input_message = {"type": "user", "content": "hi! I'm bob"}
for chunk in graph.stream({"messages": [input_message]}, {}, 
stream_mode="values"):
   chunk["messages"][-1].pretty_print()
print(results)
```

### Retrieval Augmented Generation 

Retrieval-augmented generation (RAG) is a technique that combines the strengths of retrieval-based models and generative models to achieve this goal. RAG enables agentic systems to generate more informed and contextually enriched responses.

First, we begin with a set of documents that may be useful for helping the system answer questions. These documents are broken into smaller chunks, which are then embedded using an encoder model and indexed in a vector database.

During the retrieval phase, the system searches a large corpus of documents or a vector store to find relevant pieces of information. This phase relies on efficient retrieval mechanisms to quickly identify and extract pertinent content.

During the generation phase, the retrieved information is fed into a generative foundation model, which uses this context to produce a coherent and contextually appropriate response.

By leveraging external knowledge and integrating it into the generation process, RAG enables the creation of more informed, accurate, and contextually relevant responses.

### Semantic Experience Memory 

With each user input text is turned into vector representation using an embedding model. Embedding is then used as query in vector search across all of previous interactions in memory store. Semantic experience memory allows agentic systems draw upon a broad base of knowledge but also tailor their responses and actions based on accumulated experience, leading to more adaptive and personalized behavior.

## GraphRAG 

GraphRAG is an extension of RAG model, incorporating graph based data structures to enhance the retrieval process. By utilizing graphs GraphRAG can manage and utilize complex interrelationships and dependencies between pieces of information. 

RAG approaches struggle when  

- Answers require connecting information scattered across multiple sources. 
- Queries involce summarizing higher level semantic themes across dataset
- Dataset is large, messy or organized narratively rather as discrete facts. 


### Using Knowledge Graphs

In GraphRAG, the retrieval phase does not simply pull relevant documents or text snippets; instead, it analyzes and retrieves nodes and edges from a graph that represents complex relationships and contextual information within the data.

GraphRAG consists of the following three components:

- Knowledge Graph
    - This component stores data in a graph structure where entities and their relationships are explicitly defined. Graph databases are highly efficient at managing connected data and supporting complex, multi-hop queries.

- Retrieval System
    - The retrieval system in GraphRAG is designed to efficiently query the graph database, extracting relevant subgraphs or clusters of nodes based on the input query or context.

- Generative Model
    - The generative model consumes the retrieved subgraph and uses this structured context to generate coherent, accurate, and contextually enriched responses.
