import networkx as nx

def build_knowledge_graph(docs):
    print("Building In-Memory Knowledge Graph...")
    G = nx.Graph()
    for doc in docs:
        content = doc.page_content
        lines = content.split('\n')
        for line in lines:
            if "|" in line and "Event" not in line and "---" not in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 2:
                    event = parts[0]
                    venue = parts[1]
                    G.add_node(event, type="Event")
                    G.add_node(venue, type="Venue")
                    G.add_edge(event, venue, relation="hosted_at")
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def search_graph(G, query):
    context_strings = []
    query_lower = query.lower()
    for node in G.nodes():
        if node.lower() in query_lower:
            neighbors = list(G.neighbors(node))
            if neighbors:
                context_strings.append(f"GRAPH FACT: {node} is connected to {', '.join(neighbors)}.")
    return "\n".join(context_strings)