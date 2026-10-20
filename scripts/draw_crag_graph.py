from app.graph.crag_graph import build_crag_graph

graph = build_crag_graph(
    lambda q, f: [], lambda q, d: d, lambda q, d: None, lambda q, p: q, lambda t, d: None
)
print(graph.get_graph().draw_mermaid())