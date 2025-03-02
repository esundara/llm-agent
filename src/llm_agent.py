# -*- coding: utf-8 -*-
"""
llm_agent
Runs best on Google Collab
"""

!pip install termcolor > /dev/null
!pip install langchain
!pip install openai
!pip install langchain_experimental
!pip install tiktoken
!pip install faiss-cpu==1.7.4
from datetime import datetime, timedelta
from typing import List
import math
import faiss
import os
import logging
logging.basicConfig(level=logging.ERROR)
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from termcolor import colored
from langchain_experimental.generative_agents import (

    GenerativeAgent,
    GenerativeAgentMemory,
)

os.environ["OPENAI_API"] = 'OPEN_API'

USER_NAME = "Elango. "  # name to use while calling the agent.

LLM = ChatOpenAI(max_tokens=100)  # using open ai LLM.

"""

## Implementing Generative Agent

"""

def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of  embeddings (OpenAI's are unit norm!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Creating a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )

santhosh_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8,  # using a relatively low number to show how reflection works
)

# Defining the Generative Agent: Alexis
santhosh = GenerativeAgent(
    name="santhosh",
    age=12,
    traits="curious, creative walker",  # Persistent traits of Santhosh
    status="exploring the intersection of happiness and conversations",  # Current status of Santhosh
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=santhosh_memory,
)

# The current "Summary" of a character can't be made because the agent hasn't made
# any observations yet.
print(santhosh.get_summary())

# Adding memories directly to the memory object

santhosh_observations = [
    "Santhosh likes his walk in the park",
    "Santhosh like to be always like to lead",
    "Santhosh is loud with his friends",
    "Santhosh does not think much",
    "Santhosh is is always looking for dinner",
    "Santhosh loves rides in the car",
    "Santhosh does not think about  outcomes."
    "Santhosh is always thinking about food"
]

for observation in santhosh_observations:
    santhosh.memory.add_memory(observation)



#basic description of santhosh
print(santhosh.get_summary(force_refresh=True))

"""## Interacting and Providing Context to Generative Characters

## Pre-Interview with Character

Before sending our character on their way, let's ask them a few questions.
"""

def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the Elango user interact with the agent."""
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message)[1]

interview_agent(santhosh, "What do you like to do?")

"""## Step through the day's observations."""

# Adding observations to Santhosh' memory
santhosh_observations_day = [
    "Santhosh likes his walk in the park",
    "Sanhosh like to be always like to lead",
    "Santhosh is loud with his friends",
    "Santhosh does not think much",
    "Sanhosh is is always looking for dinner",
    "Santhosh loves rides in the car",
    "Santhosh does not think about  outcomes."
    "Santhosh is always thinking about food"
]

for observation in santhosh_observations_day:
    santhosh.memory.add_memory(observation)

# Observing  Santhosh's day influences his memory and character
for i, observation in enumerate(santhosh_observations_day):
    _, reaction = santhosh.generate_reaction(observation)
    print(colored(observation, "blue"), reaction)
    if ((i + 1) % len(santhosh_observations_day)) == 0:
        print("*" * 60)
        print(
            colored(
                f"After these observations, Santhosh's summary is:\n{santhosh.get_summary(force_refresh=True)}",
                "blue",
            )
        )
        print("*" * 60)