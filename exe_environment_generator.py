from edge_cloud_env.edge_cloud import *
from edge_cloud_env.processor import *
import math
import numpy as np
from edge_cloud_env.environment import *
from edge_cloud_env.net_env import *
from edge_cloud_env.net_channel import *

def generate_exe_environment(num_edge_clouds,lis_dist_ec_grade,dic_num_ec_grade,dic_class_edge_clouds,cap_processor,dic_num_ec_class_total,
                             dic_num_ec_class_grade,bw_in_channel,min_num_channels,max_num_channels,avg_processor_err_prob,avg_bw_err_prob):
    # pass
    lis_edge_clouds=[]                                       #该执行环境中edge clouds对象的列表
    ec_idx=0
    envir=environment(num_edge_clouds,None,None,None,None,len(lis_dist_ec_grade),lis_dist_ec_grade)
    for dist_grade in lis_dist_ec_grade:             #分层（grade）生成，注：lis_dist_ec_grade中也包含距离为0的移动设备，且dic_class_edge_clouds中包含移动设备特有的processor数量
        # pass
        dic_class_ecs={}
        lis_total_pos=GetTotalPos(dist_grade,math.pi/6)                            #该层可选的位置列表（指定12个，夹角30度）
        lis_chosn_pos=[]                                                           #该层已选的位置列表
        dic_num_ec_class=dic_num_ec_class_grade[dist_grade]
        for ec in dic_class_edge_clouds:                     #遍历不同的edge cloud类型
            if dic_num_ec_class[ec]>0:                       #如果包含该类型的edge cloud
                num_processors=dic_class_edge_clouds[ec]     #该类型的edge cloud含有的processors（考虑同构）数量
                lis_processors_grade=[]                            #该grade中processors对象列表
                for _ in range(dic_num_ec_class[ec]):        #在该层（grade）逐个生成该类型的edge cloud
                    ed_cloud=edge_cloud(ec_idx,num_processors,None,None,avg_processor_err_prob,lis_dist_ec_grade.index(dist_grade))
                    # processor_id=0                           #该edge cloud中的processor id
                    pos_ec=GetRandCoord(dist_grade,lis_total_pos,lis_chosn_pos)            #均匀随机产生该ec的pos
                    lis_chosn_pos.append(pos_ec)
                    ed_cloud.Set_coord(pos_ec)
                    lis_processor_ec=[]
                    for processor_id in range(num_processors):
                        # pass
                        procr=processor(processor_id,ed_cloud,cap_processor,True,avg_processor_err_prob)     #（同一ec下的）processor具有相同的处理能力
                        lis_processor_ec.append(procr)
                    ed_cloud.Set_lis_processors(lis_processor_ec)
                    lis_edge_clouds.append(ed_cloud)
                    ec_idx+=1
    lis_ecs=sorted(lis_edge_clouds,key=lambda s:s.grade)
    envir.Set_lis_ecs(lis_ecs)                             #设置环境中的edge clouds列表

    #生成记录各个edge cloud的出错概率的列表
    lis_err_prob_op=GenerateErrProbOp(lis_ecs)
    envir.Set_lis_err_prob_op(lis_err_prob_op)

    #生成距离邻接矩阵
    dis_adj_mat=GenerateDisAdjMat(lis_ecs)
    envir.Set_dis_adj_mat(dis_adj_mat)

    #生成网络环境邻接矩阵
    net_env_adj_mat=GenerateNetEnvAdjMat(lis_ecs,dis_adj_mat,bw_in_channel,min_num_channels,max_num_channels,avg_bw_err_prob)
    envir.Set_net_env_adj_mat(net_env_adj_mat)

    #生成网络传输出错概率的邻接矩阵
    err_adj_mat=GenerateErrAdjMat(net_env_adj_mat)
    envir.Set_err_adj_mat(err_adj_mat)

    return envir





def GetTotalPos(dist_grade,angle):
    if dist_grade==0:
        return [(0,0)]
    lis_total_pos=[(dist_grade,0),(-dist_grade,0),(0,dist_grade),(0,-dist_grade)]
    x_tem=round(dist_grade*math.sin(angle),2)
    y_tem=round(dist_grade*math.cos(angle),2)
    lis_total_pos.extend([(x_tem,y_tem),(y_tem,x_tem),(y_tem,-x_tem),(x_tem,-y_tem),(-x_tem,-y_tem),(-y_tem,-x_tem),(-y_tem,x_tem),(-x_tem,y_tem)])
    return lis_total_pos

def GetRandCoord(dist_grade,lis_total_pos,lis_chosn_pos):                  #均匀随机的生成ec的部署位置
    if dist_grade==0:
        return (0,0)
    len_total=len(lis_total_pos)
    pos_idx=np.random.randint(len_total)
    while lis_total_pos[pos_idx] in lis_chosn_pos:
        pos_idx=(pos_idx+1)%len_total
    return lis_total_pos[pos_idx]

def GetSpeCoord():                                              #生成特定的部署位置
    pass

def GenerateDisAdjMat(lis_ecs):
    num_ecs=len(lis_ecs)
    dis_adj_mat=np.ones((num_ecs,num_ecs),dtype=np.int)
    for i in range(num_ecs):
        for j in range(num_ecs):
            dis_adj_mat[i][j]=lis_ecs[i].Cal_dis_with_ec(lis_ecs[j])
    return dis_adj_mat.tolist()

def GenerateNetEnvAdjMat(lis_ecs,dis_adj_mat,bw_in_channel,min_num_channels,max_num_channels,avg_bw_err_prob):
    # pass
    num_ecs=len(lis_ecs)
    max_dis=max(max(row) for row in dis_adj_mat)
    min_dis=10000
    for dis_row in dis_adj_mat:
        for dis in dis_row:
            if dis!=0 and dis<min_dis:
                min_dis=dis
    # for i in range(num_ecs):
    #     for j in range(num_ecs):
    #         if dis_adj_mat[i,j]>max_dis:
    #             max_dis
    net_env_adj_mat=np.zeros((num_ecs,num_ecs),dtype=net_env)
    ne_idx=0
    for i in range(num_ecs):
        for j in range(num_ecs):
            if i!=j:
                num_channels=min_num_channels+round((max_num_channels-min_num_channels)*((max_dis-dis_adj_mat[i][j])/max_dis))
                if min_dis==max_dis:
                    num_channels=round((max_num_channels+min_num_channels)/2)
                elif dis_adj_mat[i][j]==max_dis:
                    num_channels=min_num_channels
                elif dis_adj_mat[i][j]==min_dis:
                    num_channels=max_num_channels
                num_channels=2
                # print('num_channels: ',num_channels)
                # print('min_dis: ',min_dis,' max_dis: ',max_dis)
                # print('dis_adj_mat: ',dis_adj_mat)
                n_e=net_env(ne_idx,lis_ecs[i],lis_ecs[j],num_channels,False,None,avg_bw_err_prob)
                lis_ncs=[]
                for channel_idx in range(num_channels):
                    n_c=net_channel(channel_idx,n_e,bw_in_channel,avg_bw_err_prob)
                    lis_ncs.append(n_c)
                n_e.Set_lis_net_channels(lis_ncs)
                # n_e.lis_ear_channels=lis_ncs
                net_env_adj_mat[i,j]=n_e
                # net_env_adj_mat[j,i]=n_e
                ne_idx+=1
    return net_env_adj_mat.tolist()

def GenerateErrAdjMat(net_env_adj_mat):
    num_ecs=len(net_env_adj_mat)
    err_adj_mat=np.zeros((num_ecs,num_ecs),dtype=np.int)
    for i in range(num_ecs):
        for j in range(num_ecs):
            if i!=j:
                err_adj_mat[i][j]=net_env_adj_mat[i][j].Get_err_prob_tran()
    return err_adj_mat

def GenerateErrProbOp(lis_ecs):
    err_prob_op=[]
    for ec in lis_ecs:
        err_prob_op.append(ec.avg_err_prob)
    return err_prob_op