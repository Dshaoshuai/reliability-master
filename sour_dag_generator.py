import math
from edge_cloud_env.module import *
import random
import numpy as np

def generate_sour_dag(num_mods,num_pre_edge_md,is_stat,perc_stat_mods,avg_indegree):        #module总数，数据边总数，标识其是否为有状态，有状态的模块数占比
    # num_mods-=2                                  #修改modules总数为包含entry module和exit module
    num_layers=int(math.sqrt(num_mods))
    num_mds_in_layer=int(num_mods/num_layers)
    # num_stat_mds=num_mods*perc_stat_mods
    stat_mds_lis=generate_stat_mds_lis(num_mods,perc_stat_mods,is_stat)
    # num_pre_edge_md=round((num_edges-num_mds_in_layer-(num_mods-(num_layers-1)*num_mds_in_layer)-2)/(num_mods-num_mds_in_layer))
    # lis_nul_out_degrees_mds=[]
    module_set=[]
    # stat_md_count=0
    # mds_count=0
    # edges_count=0
    stat_sign=False
    if is_stat and random.random()<=perc_stat_mods:
        stat_sign=True
    entry_md=module(0,None,None,None,None,stat_sign,None,0)   #第0层，1个entry module，注：这个module并不是entry module，真正的entry module一定需要部署在移动端，下面的exit module也是这样
    # lis_nul_out_degrees_mds.append(entry_md)
    module_set.append(entry_md)
    md_idx=1
    for layer in range(1,num_layers+2):
        # print(layer,num_layers+1)
        # num_edge_layer=0
        if layer==1:             #第1层
            for _ in range(num_mds_in_layer):
                if md_idx in stat_mds_lis:
                    md=module(md_idx,None,None,None,None,True,None,layer)
                else:
                    md=module(md_idx,None,None,None,None,False,None,layer)
                # in_degree=0
                # while in_degree<=0:
                #     in_degree=np.random.normal(param_in_degree[0],param_in_degree[1])
                in_degree=1
                md.Set_indegree(in_degree)
                md.Add_pre_mod(entry_md)
                entry_md.Add_sub_mod(md)
                module_set.append(md)
                md_idx+=1
            # mds_count+=num_mds_in_layer
        elif layer>1 and layer<=num_layers:                 #第2层到第num_layers层
            # pass
            if layer==num_layers:                #第num_layers层
                num_mds_in_layer=num_mods-num_mds_in_layer*(num_layers-1)
            # print(num_mds_in_layer)                                                            # ***
            for i in range(1,num_mds_in_layer+1):
                # print(i)                                                                       # ***
                if md_idx in stat_mds_lis:
                    md=module(md_idx,None,None,None,None,True,None,layer)
                else:
                    md=module(md_idx,None,None,None,None,False,None,layer)
                # if i <num_mds_in_layer:                    #该层第1个到第num_mds_in_layer-1个module
                lis_pre_mds=Get_lis_pre_mds(module_set,num_pre_edge_md,md,i,num_mds_in_layer)  #得到前驱模块集合

                print('idx lis_pre_mds: ',[md.idx for md in lis_pre_mds])                                          # ***
                Update_md_pre_lis(md,lis_pre_mds)
                module_set.append(md)
                print('md_idx: ', module_set.index(md))
                md_idx+=1
        # elif layer==num_layers:                            #第num_layers层
        #     for _ in range(num_mods-num_mds_in_layer*(num_layers-1)):
        #         if md_idx in stat_mds_lis:
        #             md=module(md_idx,None,None,None,None,True,layer)
        #         else:
        #             md=module(md_idx,None,None,None,None,False,layer)
        #         module_set.append(md)
        #         md_idx+=1
        elif layer==num_layers+1:                                              #第num_layers+1层，即exit module，这个module并不是exit module，真正的exit module在生成的job中没有被包含，但是计算make-span时需要考虑进来
            # pass
            stat_sign = False
            if is_stat and random.random() <= perc_stat_mods:
                stat_sign = True
            md=module(md_idx,None,None,None,None,stat_sign,None,layer)
            lis_up_layer_mds=Get_lis_up_layer_mds(md,module_set)
            Update_md_pre_lis(md,lis_up_layer_mds)
            module_set.append(md)
    # return Sort_module_set(module_set)
    return module_set





def generate_stat_mds_lis(num_mods,prop_stat_mods,is_stat):
    stat_mds_lis=[]
    num_stat_mds=int(num_mods*prop_stat_mods)
    if num_stat_mds==0 or not is_stat:
        return []
    else:
        i=0
        a = np.random.randint(1, num_mods + 1)
        while i<num_stat_mds:
            # a=np.random.randint(1,num_mods+1)
            a=(a+1)%num_mods
            if a==0:
                a+=1
            if a not in stat_mds_lis:
                stat_mds_lis.append(a)
                i+=1
    return stat_mds_lis


def Get_lis_pre_mds(module_set,indegree,module,idx_in_layer,num_mds_in_layer):   #得到该module的前驱模块集合
    lis_up_layer_mds=[]
    lis_res_mds=[]
    md_layer=module.layer
    for md in module_set:
        if md.layer==md_layer-1:
            lis_up_layer_mds.append(md)
        elif md.layer<md_layer-1:
            lis_res_mds.append(md)
    lis_pre_mds=[]
    # print(indegree)                                      # ***
    if len(lis_res_mds)==1:                      #如果除上一层的module之外的module数量等于1，即当前层为第2层
        if random.random()<=0.5:                 #以一定的概率加入entry module作为其前驱模块
            lis_pre_mds.append(lis_res_mds[0])
            indegree-=1
    else:
        # pass
        i=0
        a = np.random.randint(len(lis_res_mds))
        while i<int(indegree/3):
            # print(i)
            # a=np.random.randint(len(lis_res_mds))
            a=(a+1)%len(lis_res_mds)
            # print(lis_res_mds[a] not in lis_pre_mds)
            if lis_res_mds[a] not in lis_pre_mds:
                lis_pre_mds.append(lis_res_mds[a])
                i+=1
        indegree-=int(indegree/3)
    # print(len(lis_res_mds),indegree)                                  # ***
    i=0
    a = np.random.randint(len(lis_up_layer_mds))
    # b = np.random.randint(len(lis_res_mds))
    # count=0
    while i<indegree:
        # a=np.random.randint(len(lis_up_layer_mds))
        # a=(a+1)%len(lis_up_layer_mds)
        # print(i,lis_up_layer_mds[a] not in lis_pre_mds)
        # if count<len(lis_up_layer_mds):
        a = (a + 1) % len(lis_up_layer_mds)
        if lis_up_layer_mds[a] not in lis_pre_mds:
            lis_pre_mds.append(lis_up_layer_mds[a])
            i+=1
            # count+=1
        # elif count>=len(lis_up_layer_mds) and count<=len(lis_up_layer_mds)+len(lis_res_mds):
        #     # b=np.random.randint(len(lis_res_mds))
        #     b=(b+1)%len(lis_res_mds)
        #     # print('***',b)
        #     if lis_res_mds[b] not in lis_pre_mds:
        #         lis_pre_mds.append(lis_res_mds[b])
        #         i+=1
        #         count+=1
        # else:
        #     print('该module的入度大于其前驱祖先modules总数，需要的上一层的indegree：',indegree,'；祖先modules总数为：',len(lis_res_mds)+len(lis_up_layer_mds))
        #     break
        # count+=1
    # if idx_in_layer==num_mds_in_layer:
    #     count=0
    #     for md in lis_up_layer_mds:
    #         if len(md.lis_sub_mods)==0 and md not in lis_pre_mds:
    #             lis_pre_mds.append(md)
    #             count+=1
    #     if count>0:
    #         print('第',md_layer,'层增加了 ',count,' 入度值。')
    return lis_pre_mds

def Update_md_pre_lis(module,lis_pre_mds):
    # pass
    for md in lis_pre_mds:
        module.Add_pre_mod(md)
        md.Add_sub_mod(module)

def Get_lis_up_layer_mds(module,module_set):
    lis_up_layer_mds=[]
    md_layer=module.layer
    for md in module_set:
        if md.layer==md_layer-1:
            lis_up_layer_mds.append(md)
    return lis_up_layer_mds


def Sort_module_set(module_set):                    #将module set进行拓扑排序
    # pass
    entry_md=module_set[0]
    sorted_md_set=[]
    sorted_md_set.append(entry_md)
    # level_tmp=[]
    i=0
    while sorted_md_set[i].lis_sub_mods:            #当添加了exit module之后结束，也是module set中的最后一个
        # pass
        sorted_md_set[i].Sort_lis_sub_mods()
        for md in sorted_md_set[i].lis_sub_mods:
            if md not in sorted_md_set:
                sorted_md_set.append(md)
        # sorted_md_set.extend(sorted_md_set[i].lis_sub_mods)
        i+=1
    return sorted_md_set
