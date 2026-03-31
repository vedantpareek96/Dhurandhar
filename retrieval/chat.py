"""
Interactive CLI chat interface for querying the data.gov.in knowledge graph.

Usage:
    python -m retrieval.chat
    python -m retrieval.chat --mode cypher     # use LangChain Cypher chain
    python -m retrieval.chat --mode hybrid     # use hybrid vector+LLM (default)
    python -m retrieval.chat --top-k 12
"""
import argparse
import logging
import sys

logging.basicConfig(level=logging.WARNING)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║   data.gov.in Knowledge Graph — Natural Language Chat    ║
║   Type your question, or 'exit' / 'quit' to stop.       ║
║   Commands: /cypher <q>  /vector <q>  /stats  /help      ║
╚══════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Available commands:
  /stats           — show graph statistics
  /cypher <q>      — use LangChain Cypher chain for this query
  /vector <q>      — show raw vector search results
  /help            — show this help
  exit / quit      — exit

Default mode: hybrid (vector search + LLM synthesis)

Example questions:
  Which sectors have the most datasets?
  Show datasets about maternal health at district level
  What agriculture data is available from Ministry of Agriculture?
  Which datasets are updated monthly and available as CSV?
"""


def _stats(driver) -> str:
    with driver.session() as session:
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY count DESC
        """)
        rows = [(r["label"], r["count"]) for r in result]

    lines = ["Graph Statistics:"]
    for label, count in rows:
        lines.append(f"  {label:<20} {count:>8,} nodes")
    return "\n".join(lines)


def main(mode: str = "hybrid", top_k: int = 8):
    from graph.loader import get_driver

    print(BANNER)

    driver = get_driver()

    # Lazy imports to avoid slow startup
    if mode == "cypher":
        from retrieval.cypher_chain import build_cypher_chain
        chain = build_cypher_chain(verbose=False)
        print(f"Mode: LangChain GraphCypherQAChain\n")
    else:
        from retrieval.hybrid_retriever import ask as hybrid_ask
        import anthropic
        from config.settings import ANTHROPIC_API_KEY
        llm_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        print(f"Mode: Hybrid (vector search + Claude synthesis), top_k={top_k}\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            if user_input == "/help":
                print(HELP_TEXT)
                continue

            if user_input == "/stats":
                print(_stats(driver))
                continue

            # /vector <query>
            if user_input.startswith("/vector "):
                from retrieval.vector_search import semantic_search, format_search_results
                q = user_input[8:].strip()
                hits = semantic_search(q, top_k=top_k, driver=driver)
                print(format_search_results(hits))
                continue

            # /cypher <query>
            if user_input.startswith("/cypher "):
                from retrieval.cypher_chain import build_cypher_chain, ask as cypher_ask
                q = user_input[8:].strip()
                local_chain = build_cypher_chain(verbose=True)
                result = cypher_ask(q, chain=local_chain)
                print(f"\nCypher: {result['cypher']}")
                print(f"Answer: {result['answer']}\n")
                continue

            # Default mode
            print("Assistant: ", end="", flush=True)
            try:
                if mode == "cypher":
                    from retrieval.cypher_chain import ask as cypher_ask
                    result = cypher_ask(user_input, chain=chain)
                    print(result["answer"])
                    if result.get("cypher"):
                        print(f"  [Cypher: {result['cypher'][:100]}...]")
                else:
                    result = hybrid_ask(
                        user_input, top_k=top_k,
                        driver=driver, client=llm_client
                    )
                    print(result["answer"])
                    if result.get("context_datasets"):
                        used = ", ".join(filter(None, result["context_datasets"][:3]))
                        print(f"  [Based on: {used}{'...' if len(result['context_datasets']) > 3 else ''}]")
            except Exception as e:
                print(f"Error: {e}")

            print()

    finally:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with the data.gov.in knowledge graph")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "cypher"],
                        help="Query mode: hybrid (default) or cypher")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Number of datasets to retrieve for context (default: 8)")
    args = parser.parse_args()
    main(mode=args.mode, top_k=args.top_k)
