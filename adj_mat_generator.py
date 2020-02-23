import sour_dag_generator as sdg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

def generate_adj_mat(module_set):             #输入排序后的module_set，输出该job的邻接矩阵，标志所有modules是否有状态的列表
    dim=len(module_set)                       #此时输出的adj_mat的维度为总modules数（args.num_mods+2）
    adj_mat=np.zeros((dim,dim),dtype=np.int)
    i=0
    for md in module_set:
        if md.lis_sub_mods:                   #exit module的出度为0，故到其时结束
            for sub_md in md.lis_sub_mods:
                # pass
                adj_mat[i][module_set.index(sub_md)]=1
        i+=1
    lis_stat_mds=np.zeros(dim,dtype=np.int)
    for i in range(dim):
        if module_set[i].is_stateful:
            lis_stat_mds[i]=1

    if is_dag(dim,adj_mat):
        return adj_mat,lis_stat_mds
    else:
        print('error! 该job的生成不符合dag要求！')
        print(adj_mat)
        return None,None

def is_dag(num_nodes, adj_mat):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i, j] == 1:
                G.add_edge(i, j)
    return nx.is_directed_acyclic_graph(G)

def generate_dag(adj_mat):
    num_nodes=adj_mat.shape[0]
    G=nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i,j]==1:
                G.add_edge(i,j)
    return G



def cal_avg_indegrees(num_mods,num_edges):
    num_layers=int(math.sqrt(num_mods))
    num_mds_in_layer=int(num_mods/num_layers)
    num_pre_edge_md = round((num_edges - num_mds_in_layer - (num_mods - (num_layers - 1) * num_mds_in_layer) - 2) / (
                num_mods - num_mds_in_layer))
    return num_pre_edge_md

if __name__=="__main__":
    job_0=sdg.generate_sour_dag(10,3,True,0.5,2)              #num_mds不包含entry module和exit module，故总数应该加2，同理，在配置文件中的modules总数也是实际的modules总数少2
    adj_mat,_=generate_adj_mat(job_0)
    print(adj_mat)
    G=generate_dag(adj_mat)
    nx.spring_layout(G)
    nx.draw_networkx(G)
    plt.show()




