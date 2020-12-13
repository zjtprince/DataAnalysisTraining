import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from collections import defaultdict

aliases = pd.read_csv("~/Documents/PageRank-master/input/Aliases.csv")
persons = pd.read_csv("~/Documents/PageRank-master/input/Persons.csv")
emails = pd.read_csv("~/Documents/PageRank-master/input/Emails.csv")

print(emails.info())

features = ['MetadataFrom', 'MetadataTo']

print(emails[features].head())

alias_person_map = {}
person_id_name_map = {}

for _, row in aliases.iterrows():
    alias_person_map[row['Alias']] = row['PersonId']

# print(alias_person_map)


for _, row in persons.iterrows():
    person_id_name_map[row['Id']] = row['Name']


def unify_name(name):
    name = str(name).lower()

    name = name.replace(",", "").split("@")[0]

    if name in alias_person_map.keys():
        return person_id_name_map[alias_person_map[name]]
    return name


def show_graph(graph, layout='spring_layout'):
    # 使用 Spring Layout 布局，类似中心放射状
    if layout == 'circular_layout':
        positions = nx.circular_layout(graph)
    else:
        positions = nx.spring_layout(graph)
    # 设置网络图中的节点大小，大小与 pagerank 值相关，因为 pagerank 值很小所以需要 *20000
    nodesize = [x['pagerank'] * 20000 for v, x in graph.nodes(data=True)]
    # 设置网络图中的边长度
    edgesize = [np.sqrt(e[2]['weight']) for e in graph.edges(data=True)]
    # 绘制节点
    nx.draw_networkx_nodes(graph, positions, node_size=nodesize, alpha=0.4)
    # 绘制边
    nx.draw_networkx_edges(graph, positions, width=edgesize, alpha=0.2)
    # 绘制节点的 label
    nx.draw_networkx_labels(graph, positions, font_size=10)
    # 输出希拉里邮件中的所有人物关系图
    plt.show()


emails['MetadataFrom'] = emails['MetadataFrom'].apply(unify_name)
emails['MetadataTo'] = emails['MetadataTo'].apply(unify_name)

print(emails[features].head())

email_count = defaultdict(list)

for row in zip(emails.MetadataFrom, emails.MetadataTo):
    edge = (row[0], row[1])
    if edge not in email_count:
        email_count[edge] = 1
    else:
        email_count[edge] += 1

g = nx.DiGraph()

edge_weights = [(key[0], key[1], value) for key, value in email_count.items()]

g.add_weighted_edges_from(edge_weights)

pagerank = nx.pagerank(g)

nx.set_node_attributes(g, values=pagerank, name='pagerank')

show_graph(g)

threshold = 0.005

small_graph = g.copy()

for index, value in g.nodes(data=True):
    if value['pagerank'] < threshold:
        small_graph.remove_node(index)

show_graph(small_graph, 'circular_layout')
