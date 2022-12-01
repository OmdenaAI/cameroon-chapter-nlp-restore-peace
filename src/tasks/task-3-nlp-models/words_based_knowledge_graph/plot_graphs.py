import numpy as np
import pandas as pd

from pyvis.network import Network


def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=0))
    return exp_values/np.sum(exp_values, axis=0)

def get_softmax_norm(x, slope=1.0, scale=200):                     
    return scale*softmax(np.log(x) + 1.0)

def create_graph(df, edges_norm=False, min_degree_node=None):
    net = Network()
    
    df = df.groupby(['src', 'dst']).sum()
    display(df.head())
    list_index = list(dict(df.index).keys())    
    if min_degree_node is not None:
        series_count_src = df.reset_index()['src'].value_counts()
        list_index = series_count_src[series_count_src >= min_degree_node].index
            
    if edges_norm:
        df['edge_norm'] =  get_softmax_norm(df['edge'])
        
    print('lenght src nodes:', len(list_index), 'min_degree_node:', min_degree_node)
    print(df.shape)
        
    for index in list_index:            
        #print(index)
        df_graph_plot = df.loc[index].reset_index()
        df_graph_plot['src'] = index
        net.add_nodes(list(df_graph_plot['dst'].values) + [df_graph_plot['src'].iloc[0]])
        if edges_norm:
            net.add_edges(list(df_graph_plot[['src', 'dst', 'edge_norm']].values))
        else:
            net.add_edges(list(df_graph_plot[['src', 'dst', 'edge']].values))

    return net

def plot_graph(net, name_file):
    net.toggle_physics(True)
    net.set_options("""
    const options = {
      "physics": {
        "forceAtlas2Based": {
          "springLength": 100,
          "springConstant": 0.085,
          "damping": 0.41,
          "avoidOverlap": 1
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      },
      "nodes": {    
        "font": {
          "size": 40,
          "face": "verdana"
        }    
      }
    }
    """)
    # net.show_buttons(filter_=['nodes'])
    net.show(name_file+'.html')
