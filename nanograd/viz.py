from graphviz import Digraph

def trace(root):
    nodes = set()
    edges = set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges

def draw_graph(root, format = 'svg'):
    nodes, edges = trace(root)
    dot = Digraph(format = format, graph_attr = {'rankdir': 'LR'})
    
    for n in nodes:
        uid = str(id(n))
        label = f"{n._op} | val = {n.data:.4f} | grad = {n.grad:.4f}"
        dot.node(name = uid, label = label, shape = 'record')

    for child, parent in edges:
        dot.edge(str(id(child)), str(id(parent)))

    return dot