"""
Small test runner to execute the compiled graph on a sample question.

This script calls `run_graph` from the `graph` package built earlier and
prints the resulting generation and important state fields. It is intended
for quick manual verification after ingesting a repository into Pinecone.

Usage:
    python run_graph_test.py
"""

from graph import run_graph

# sample question about the Flask repository (assumes pallets/flask was ingested)
QUESTION = "How does Flask route HTTP requests to view functions?"
NAMESPACE = "pallets_flask"  # replace with the actual namespace used during ingestion


def main() -> None:
    """Run the graph and print key outputs for inspection."""

    print(f"[TEST] Running graph for question: {QUESTION}")
    final_state = run_graph(QUESTION, NAMESPACE)

    # print summary of important fields from the final state
    print("\n=== Graph Final State Summary ===")
    print(f"Generation:\n{final_state.get('generation')}\n")
    print(f"Loop count: {final_state.get('loop_count')}")
    print(f"Retrieval score: {final_state.get('retrieval_score')}")
    print(f"Source files used: {final_state.get('source_files')}")


if __name__ == '__main__':
    main()
