"""
LangChain GraphCypherQAChain: natural language → Cypher → answer.
Uses OpenAI GPT-4o and the Neo4j graph schema.
"""
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY, REASONING_MODEL


def build_cypher_chain(verbose: bool = False) -> GraphCypherQAChain:
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
    llm = ChatOpenAI(model=REASONING_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    return GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=verbose,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
    )


def ask(question: str, chain: GraphCypherQAChain = None, verbose: bool = False) -> dict:
    if chain is None:
        chain = build_cypher_chain(verbose=verbose)
    result = chain.invoke({"query": question})
    cypher = ""
    for step in (result.get("intermediate_steps") or []):
        if "query" in step:
            cypher = step["query"]
            break
    return {"answer": result.get("result", ""), "cypher": cypher}
