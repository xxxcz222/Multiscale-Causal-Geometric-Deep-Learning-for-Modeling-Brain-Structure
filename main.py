import networkx as nx
import matplotlib.pyplot as plt

# 创建图
G = nx.Graph()

# 添加节点及其坐标

positions = {
    0: (0.7, 1.0),
    1: (0.4, 0.3),
    2: (0.7, 0.6),
    3: (0.2, 0.7),
    4: (0.4, 0.2),
}

positions = {
    0: (0.7, 1.0),
    1: (0.1, 0.3),
    2: (0.7, 0.6),
    3: (0.2, 0.5),
    4: (0.9, 0.2),
}

positions = {
    0: (0.7, 1.0),
    1: (0.1, 0.3),
    2: (0.7, 0.6),
    3: (0.2, 0.5),
    4: (0.9, 0.2),
    5: (0.7, 1.0),
    6: (0.4, 0.3),
    7: (0.7, 0.6),
    8: (0.2, 0.7),
    9: (0.4, 0.2),
}

G.add_nodes_from(positions.keys())

# 添加边（不规则连接）
edges = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (1, 2),
    (5, 8),
    (5, 7),
    (6, 9),
    (6, 7),
]
G.add_edges_from(edges)

# 绘图
plt.figure(figsize=(6, 6))
nx.draw(
    G,
    pos=positions,
    with_labels=False,
    node_size=300,
    node_color='black',
    edge_color='black',
    width=2
)
plt.show()