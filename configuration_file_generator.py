import sour_dag_generator as sdg
import adj_mat_generator as amg
import numpy as np
import math
from params import args

def generate_configuration_file(num_mods,num_edges,num_pre_edge_md,is_stat,perc_stat_mds,avg_indegrees,total_stat_volume,avg_stat_volume,
                                total_mds_op_volume,avg_md_op_volume,total_edges_data_volume,avg_edge_data_volume):
    dic_configuration_file = {'adj_mat': None, 'is_state': is_stat, 'lis_is_stat_mds': None, 'perc_stat_mds': perc_stat_mds,
                              'num_stat_mds': None,
                              'total_stat_volume': None, 'avg_stat_volume': avg_stat_volume, 'num_mds': num_mods,
                              'total_mds_op_volume': None, 'avg_md_op_volume': avg_md_op_volume,
                              'num_edges': None, 'num_pre_edge_md':num_pre_edge_md,'total_edges_data_volume': None, 'avg_edge_data_volume': avg_edge_data_volume,
                              'avg_md_indegrees': avg_indegrees} #邻接矩阵；标志job是否有状态；modules状态标志列表；（如果该job有状态）其有状态的modules占的比例；
    #有状态的modules的数量；状态量总量的大小；平均状态量的大小；模块数量；modules总运算量大小；module平均运算量大小；数据边总数；每条数据边上数据量的平均大小；每个module的平均入度
    job=sdg.generate_sour_dag(num_mods,num_pre_edge_md,is_stat,perc_stat_mds,avg_indegrees)
    adj_mat,lis_stat_mds=amg.generate_adj_mat(job)
    num_edges=cal_total_edges(num_mods,num_pre_edge_md)
    dic_configuration_file['num_edges']=num_edges
    dic_configuration_file['adj_mat']=adj_mat
    dic_configuration_file['lis_is_stat_mds']=lis_stat_mds
    dic_configuration_file['total_stat_volume']=sum(lis_stat_mds)*avg_stat_volume
    dic_configuration_file['total_mds_op_volume']=(num_mods+2)*avg_md_op_volume
    dic_configuration_file['total_edges_data_volume']=num_edges*avg_edge_data_volume
    if is_stat:
        dic_configuration_file['num_stat_mds']=sum(lis_stat_mds)
    else:
        dic_configuration_file['num_stat_mds']=0
    # np.save('C:/Users/25714/PycharmProjects/reliability/job_dags/10/{}.npy'.format(total_mds_op_volume,total_edges_data_volume),dic_configuration_file)
    return dic_configuration_file

def cal_total_edges(num_nodes,avg_indegree):           #num_nodes不考虑entry module和exit module
    # pass
    num_layers=int(math.sqrt(num_nodes))
    num_mods_in_layer=int(num_nodes/num_layers)
    return 2+num_mods_in_layer+(num_layers-1)*num_mods_in_layer*avg_indegree+num_nodes-(num_layers-1)*num_mods_in_layer

def config_files_generator():
    # pass
    # save_path='C:/Users/25714/PycharmProjects/reliability/job_dags/'+str(args.num_mods)+'/'+str(args.num_mods)+'_'
    save_path = 'D:/reliability-master/reliability-master/job_dags/' + str(args.num_mods) + '/' + str(args.num_mods) + '_'
    is_stat=True
    for i in range(10):
        if i>=5:
            is_stat=False
        dic_configuration_file=generate_configuration_file(args.num_mods,None,args.num_pre_edge_md,is_stat,args.perc_stat_mds,args.avg_indegrees,None,args.avg_stat_volume
                                                               ,None,args.avg_md_op_volume,None,args.avg_edge_data_volume)
        np.save(save_path+'{}.npy'.format(i),dic_configuration_file)

if __name__=="__main__":
    # generate_configuration_file(10,None,3,True,0.5,2,10,2,48,4,63,3)
    # # print('done!')
    for i in range(10):
        # pt_0='C:/Users/25714/PycharmProjects/reliability/job_dags/10/10_3.npy'
        pt_0 = './job_dags/10/10_'+str(i)+'.npy'
        config_file = np.load(pt_0, allow_pickle=True).item()
        # #print(config_file['adj_mat'],config_file['is_state'],config_file['lis_is_stat_mds'],config_file['perc_stat_mds'],
        #       #config_file['num_stat_mds'],config_file['total_stat_volume'],config_file['avg_stat_volume'],config_file['num_mds'],config_file['total_mds_op_volume'],
        #       #config_file['avg_md_op_volume'],config_file['num_edges'],config_file['total_edges_data_volume'],config_file['avg_edge_data_volume'],config_file['avg_md_indegrees'])
        print(config_file['adj_mat'],config_file['num_mds'])
    # config_files_generator()
    # print(args.dic_num_ec_class_grade)