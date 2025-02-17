import networkx as nx
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

# Given context of triplets
context = """
Nobel Prize in Literature | award reverse | G. K. Chesterton
Nobel Prize in Literature | award reverse | J. M. Coetzee
Thomas Mann | influenced reverse | Richard Wagner
Nobel Prize in Literature | award reverse | Aldous Huxley
Nobel Prize in Literature | award winner | Doris Lessing
Thomas Mann | influenced | Franz Kafka
Nobel Prize in Literature | award winner | Henri Bergson
Thomas Mann | influenced reverse | Sigmund Freud
Nobel Prize in Literature | award winner | Seamus Heaney
Thomas Mann | award nominee reverse | Nobel Prize in Literature
Nobel Prize in Literature | award reverse | Toni Morrison
Thomas Mann | influenced reverse | Arthur Schopenhauer
Thomas Mann | influenced | Joseph Campbell
Thomas Mann | influenced reverse | Herman Melville
Thomas Mann | influenced reverse | Edgar Allan Poe
Nobel Prize in Literature | award nominee | G. K. Chesterton
Nobel Prize in Literature | award reverse | Graham Greene
Nobel Prize in Literature | award nominee | Aldous Huxley
Nobel Prize in Literature | award reverse | Seamus Heaney
Thomas Mann | influenced | William Faulkner
Thomas Mann | award | Nobel Prize in Literature
Thomas Mann | influenced reverse | Leo Tolstoy
Thomas Mann | influenced by reverse | Franz Kafka
Nobel Prize in Literature | award reverse | T. S. Eliot
Nobel Prize in Literature | award nominee | T. S. Eliot
Thomas Mann | influenced by reverse | Joseph Campbell
Nobel Prize in Literature | award nominee | Henrik Ibsen
Nobel Prize in Literature | award reverse | Doris Lessing
Thomas Mann | influenced reverse | Hermann Hesse
Thomas Mann | influenced | Hermann Hesse
Nobel Prize in Literature | award winner | Samuel Beckett
Thomas Mann | influenced reverse | Fyodor Dostoyevsky
Nobel Prize in Literature | award reverse | Henrik Ibsen
Thomas Mann | influenced by reverse | William Faulkner
Thomas Mann | influenced reverse | Arnold Schoenberg
Nobel Prize in Literature | award reverse | Henry James
Thomas Mann | award | National Book Award for Fiction
Nobel Prize in Literature | award reverse | Samuel Beckett
Nobel Prize in Literature | award nominee | Henry James
Nobel Prize in Literature | award winner | Toni Morrison
Nobel Prize in Literature | award nominee | Graham Greene
Nobel Prize in Literature | award reverse | T. S. Eliot
Nobel Prize in Literature | award reverse | Henri Bergson
Thomas Mann | influenced by reverse | Hermann Hesse
Nobel Prize in Literature | award reverse | Henri Bergson
Nobel Prize in Literature | award nominee | Henri Bergson
Nobel Prize in Literature | award winner | J. M. Coetzee
Nobel Prize in Literature | award winner | T. S. Eliot
"""

G = nx.DiGraph()

for line in context.strip().split('\n'):
    source, relation, target = line.split(' | ')
    G.add_edge(source, target, label=relation)


plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)  
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, arrows=True)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Neighborhood graph")
plt.show()
