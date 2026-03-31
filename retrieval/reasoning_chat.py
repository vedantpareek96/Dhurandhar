"""
Interactive reasoning chat over data.gov.in knowledge graph.
Each question gets a full structured analysis: datasets found,
coverage, gaps, and a concrete action plan.

Usage:
    python -m retrieval.reasoning_chat
    python -m retrieval.reasoning_chat --top-k 15
"""
import argparse
import json
import logging
import sys

logger = logging.basicConfig(level=logging.WARNING)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   data.gov.in Reasoning Engine                               ║
║   Ask any research question — I'll tell you:                 ║
║     • Which datasets are relevant                            ║
║     • Whether your analysis is feasible                      ║
║     • What's missing and what to do next                     ║
║                                                              ║
║   Commands: /stats   /search <q>   /help   exit              ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Commands:
  /stats           — show graph statistics (node counts by label)
  /search <query>  — raw vector similarity search
  /json            — toggle JSON output mode (shows raw ReasoningResult)
  /help            — this help
  exit / quit      — exit

Example questions:
  How is agricultural productivity related to water availability across districts?
  Which states have poor health infrastructure based on government data?
  Can I study the impact of education spending on literacy rates over time?
  What data exists about urban air quality and its health effects?
  Is there data to analyze employment patterns by gender across states?
"""


def _stats(driver) -> str:
    with driver.session() as session:
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY count DESC
        """)
        rows = [(r["label"], r["count"]) for r in result]
    lines = ["\nGraph Statistics:"]
    for label, count in rows:
        lines.append(f"  {label:<22} {count:>10,} nodes")
    return "\n".join(lines)


def _render(result, json_mode: bool = False):
    """Pretty-print a ReasoningResult."""
    if json_mode:
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), indent=2, default=str))
        return

    # ── Intent ──────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  INTENT EXTRACTED")
    print(f"{'─'*62}")
    print(f"  Topics     : {', '.join(result.intent.topics) or '—'}")
    print(f"  Sectors    : {', '.join(result.intent.sectors) or '—'}")
    print(f"  Metrics    : {', '.join(result.intent.metrics) or '—'}")
    print(f"  Granularity: {result.intent.granularity}")
    print(f"  Time range : {result.intent.time_range}")
    print(f"  Analysis   : {result.intent.analysis_type}")

    # ── Datasets found ──────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  DATASETS FOUND ({len(result.matched_datasets)})")
    print(f"{'─'*62}")
    for i, d in enumerate(result.matched_datasets[:8], 1):
        gran = ", ".join(d.granularity) or "unknown"
        fmts = ", ".join(d.formats) or "unknown"
        score = f"{d.relevance_score:.2f}"
        print(f"  {i}. {d.title[:55]}")
        print(f"     Sector: {d.sector} | Ministry: {(d.ministry or '?')[:40]}")
        print(f"     Granularity: {gran} | Formats: {fmts}")
        print(f"     Records: {d.total_count:,} | Updated: {d.updated} | Score: {score}")
        if d.api_url:
            print(f"     API: {d.api_url}")
        print()

    # ── Coverage ────────────────────────────────────────
    print(f"{'─'*62}")
    print(f"  COVERAGE ASSESSMENT")
    print(f"{'─'*62}")
    gran_icon = "✅" if result.coverage.granularity_match else "⚠️ "
    print(f"  {gran_icon} {result.coverage.granularity_note}")
    print(f"  Freshness      : {result.coverage.freshness}")
    print(f"  Data volume    : {result.coverage.record_volume}")
    print(f"  Formats        : {', '.join(result.coverage.formats_available) or 'unknown'}")
    print(f"  Linkable on    : {', '.join(result.coverage.linkable_dimensions) or '—'}")

    # ── Feasibility ─────────────────────────────────────
    icon = {"YES": "✅", "PARTIAL": "⚠️ ", "NO": "❌"}[result.feasibility]
    print(f"\n{'─'*62}")
    print(f"  FEASIBILITY: {icon} {result.feasibility}")
    print(f"{'─'*62}")
    print(f"  {result.feasibility_note}")

    # ── Gaps ────────────────────────────────────────────
    if result.gaps:
        print(f"\n{'─'*62}")
        print(f"  GAPS & MISSING DATA")
        print(f"{'─'*62}")
        for g in result.gaps:
            print(f"  ⚠️  {g}")

    # ── Next steps ──────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  NEXT STEPS")
    print(f"{'─'*62}")
    for i, step in enumerate(result.next_steps, 1):
        print(f"  {i}. {step}")

    # ── API calls ───────────────────────────────────────
    if result.api_calls:
        print(f"\n{'─'*62}")
        print(f"  READY-TO-USE API CALLS")
        print(f"{'─'*62}")
        for call in result.api_calls:
            print(f"  Dataset: {call['dataset'][:55]}")
            params = "&".join(f"{k}={v}" for k, v in call["params"].items())
            print(f"  GET {call['url']}?{params}")
            print(f"  Note: {call['note']}")
            print()

    # ── Narrative ───────────────────────────────────────
    print(f"{'─'*62}")
    print(f"  ANALYSIS SUMMARY")
    print(f"{'─'*62}")
    for line in result.narrative.split(". "):
        line = line.strip()
        if line:
            print(f"  {line}.")
    print(f"{'─'*62}\n")


def main(top_k: int = 12):
    from graph.loader import get_driver
    from retrieval.reasoner import reason
    from retrieval.vector_search import semantic_search

    print(BANNER)
    driver = get_driver()
    json_mode = False

    try:
        while True:
            try:
                user_input = input("Question: ").strip()
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

            if user_input == "/json":
                json_mode = not json_mode
                print(f"JSON mode: {'ON' if json_mode else 'OFF'}")
                continue

            if user_input.startswith("/search "):
                query = user_input[8:].strip()
                hits = semantic_search(query, top_k=top_k, driver=driver)
                print(f"\nTop {len(hits)} matches for '{query}':")
                for i, h in enumerate(hits, 1):
                    gran = ", ".join(h.get("granularity") or []) or "?"
                    print(f"  {i}. [{h.get('sector','?')}] {h.get('title','?')}")
                    print(f"     Ministry: {h.get('ministry','?')} | Granularity: {gran} | Score: {h.get('score',0):.3f}")
                print()
                continue

            # ── Main: full reasoning ─────────────────────────
            print("\nAnalyzing...\n")
            try:
                result = reason(user_input, driver=driver, top_k=top_k)
                _render(result, json_mode=json_mode)
            except Exception as e:
                print(f"Error: {e}")
                logger.exception(e)

    finally:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reasoning chat over data.gov.in knowledge graph")
    parser.add_argument("--top-k", type=int, default=12,
                        help="Max datasets to retrieve per query (default: 12)")
    args = parser.parse_args()
    main(top_k=args.top_k)
