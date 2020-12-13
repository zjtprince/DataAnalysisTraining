import networkx as nx


def test_page_rank():
    G = nx.DiGraph()
    edges = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "A"), ("B", "D"), ("C", "A"), ("D", "B"), ("D", "C")]

    for edge in edges:
        G.add_edge(edge[0], edge[1])
    pagerank_list = nx.pagerank(G, alpha=0.85)
    print(pagerank_list)

test_page_rank()

def test_rank_leak():
    G = nx.DiGraph()
    edges = [("A", "B"), ("B", "D"), ("D", "A"), ("D", "C")]

    for edge in edges:
        G.add_edge(edge[0], edge[1])
    pagerank_list = nx.pagerank(G, alpha=1)
    print(pagerank_list)

# test_rank_leak()

def test_rank_sink():
    G = nx.DiGraph()
    edges = [("A", "B"), ("B", "D"), ("D", "A"), ("C", "A"),("C","D")]

    for edge in edges:
        G.add_edge(edge[0], edge[1])
    pagerank_list = nx.pagerank(G, alpha=0.85)
    print(pagerank_list)

# test_rank_sink()