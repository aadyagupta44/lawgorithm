"""
Quick test to verify the graph runs end to end.
Usage: python test_graph.py
"""

from graph import run_graph

# test with a question about flask
# using the namespace we created during ingestion
result = run_graph(
    question="What is flask and how do I create a basic app?",
    namespace="realpython_flask_boilerplate"
)

print("\n========================================")
print("FINAL ANSWER:")
print("========================================")
print(result.get("generation", "No answer generated"))
print("\nSource files used:")
for f in result.get("source_files", []):
    print(f"  - {f}")
print(f"\nLoop count: {result.get('loop_count', 0)}")
print("========================================")
