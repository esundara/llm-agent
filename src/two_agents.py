# -*- coding: utf-8 -*-
"""
two-agents.ipynb
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

os.environ["OPENAI_API_KEYYYYYYYYYY"] = ''

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

husband_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8,  # using a relatively low number to show how reflection works
)

# Defining the Generative Agent: Alexis
husband = GenerativeAgent(
    name="Husband",
    age=35,
    traits="hard working, hacker",  # Persistent traits of XY
    status="positivity and effort ",  # Current status of XY
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=husband_memory,
)


# The current "Summary" of a character can't be made because the agent hasn't made
# any observations yet.
print(husband.get_summary())

# Adding memories directly to the memory object

husband_observations = [
    "Husband like to watch movies",
    "Husband likes to chat with friends  ",
    "Husband likes to code ",
     "Husband likes to eat sambar ",
    "Husband likes to fix equipment ",

]

for observation in husband_observations:
    husband.memory.add_memory(observation)


#basic description of santhosh
print(husband.get_summary(force_refresh=True))

"""## Interacting and Providing Context to Generative Characters

## Pre-Interview with Character

Before sending our character on their way, let's ask them a few questions.
"""

def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the XY user interact with the agent."""
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message)[1]

interview_agent(husband, "What do you like to do?")

"""## Step through the day's observations."""

# Adding observations to Santhosh' memory
husband_observations_day = [
    "Husband likes his walk in the park",
    "Husband like to be always like to lead",
    "Husband likes boring songs ",
    "Husband likes to be involved",

]

for observation in husband_observations_day:
    husband.memory.add_memory(observation)

# Observing  Husband's day influences his memory and character
for i, observation in enumerate(husband_observations_day):
    _, reaction = husband.generate_reaction(observation)
    print(colored(observation, "blue"), reaction)
    if ((i + 1) % len(husband_observations_day)) == 0:
        print("*" * 60)
        print(
            colored(
                f"After these observations, Husband's summary is:\n{husband.get_summary(force_refresh=True)}",
                "blue",
            )
        )
        print("*" * 60)

wife_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8,  # using a relatively low number to show how reflection works
)

# Defining the Generative Agent: Wife
wife = GenerativeAgent(
    name="Wife",
    age=30,
    traits="practical and helpful",  # Persistent traits of Wife
    status="positivity and practical ",  # Current status of Wife
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=wife_memory,
)


# The current "Summary" of a character can't be made because the agent hasn't made
# any observations yet.
print(wife.get_summary())

# Adding memories directly to the memory object

wife_observations = [
    "Wife likes to watch movies ",
       "Wife likes to chat with relatives ",
     "Wife likes TO be happy ",
    "Wife likes cold weather ",
    "Wife likes to watch tamil languageprogram Neeya Naana on TV ",


]

for observation in wife_observations:
    wife.memory.add_memory(observation)

 #basic description of XX
print(wife.get_summary(force_refresh=True))

def run_conversation(agents: List[GenerativeAgent], initial_observation: str) -> None:
    """Runs a conversation between agents."""
    _, observation = agents[1].generate_reaction(initial_observation)
    print(observation)
    max_turns = 3
    turns = 0
    while turns<=max_turns:
        break_dialogue = False
        for agent in agents:
            stay_in_dialogue, observation = agent.generate_dialogue_response(
                observation
            )
            print(observation)
            # observation = f"{agent.name} said {reaction}"
            if not stay_in_dialogue:
                break_dialogue = True
        if break_dialogue:
            break
        turns += 1

agents = [husband, wife]
run_conversation(
    agents,
    "Wife said: Hey Husband, I've been trying to help one of the relatives with a question on AI can you suggest?",
)

interview_agent(husband, "How was your conversation with Wife ?")

interview_agent(wife, "How was your conversation with Husband ?")