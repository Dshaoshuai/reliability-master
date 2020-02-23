import numpy as np
from params import *

class JobDAG(object):
    def __init__(self,idx,num_mds,is_stateful,lis_mds,avg_md_op_amount,ori_adj_mat,prop_stat,
                 avg_stat_amount,lis_sig_md_stat,reliability_req,release_time,num_edges,ori_total_edges_data_volume,ori_avg_edge_data_volume,avg_md_indegrees,num_stat_mds,total_stat_amount,total_md_op_amount):
        self.idx=idx                                    #job id
        self.num_mds=num_mds                            #该job的模块数量
        self.is_stateful=is_stateful                    #标识该job是否为有状态
        if lis_mds:
            self.lis_mds=lis_mds                                          #该job的模块列表
        else:
            self.lis_mds=[]
        self.avg_md_op_amount=avg_md_op_amount                            #该job中module的平均运算量大小
        self.total_md_op_amount=total_md_op_amount                        #该job中module的总运算量
        # if ori_adj_mat!=None:
        self.ori_adj_mat=ori_adj_mat                                  #该job的原始邻接矩阵
        # else:
        #     self.ori_adj_mat=None

        self.avg_md_indegrees=avg_md_indegrees                            #每个module的平均入度

        self.ori_num_edges=num_edges                                      #原始数据边总数，原始的含义：不考虑冗余和重传的数量
        self.ori_total_edges_data_volume=ori_total_edges_data_volume      #原始数据边的数据总量
        self.ori_avg_edge_data_volume=ori_avg_edge_data_volume            #原始每条数据边的平均数据量
        self.ori_adj_mat_edges=None                                       #记录数据流边对象的原始邻接矩阵
        self.scal_adj_mat_edges=None                                      #扩展后的记录数据流边对象的邻接矩阵，[冗余模块数量 * 冗余模块数量]，元素为边对象的列表或None

        self.prop_stat=prop_stat                                      #如果为有状态的job，其有状态的模块所占的比例
        self.num_stat_mds=num_stat_mds                                #有状态的modules的数量
        self.avg_stat_amount=avg_stat_amount                          #有状态的module所具有的平均状态量的大小
        self.total_stat_amount=total_stat_amount                      #该job中状态量的总量
        self.lis_sig_md_stat=lis_sig_md_stat                          #标志所有模块是否有状态的0-1列表
        self.reliability_req=reliability_req                          #该job可靠性relability的需求
        self.release_time=release_time                                #该job的释放时间，即到达时间

        self.res_num_mds=num_mds                                         #该job剩余module的数量，初始化为总module的数量
        self.res_md_op_amound=-1                                         #该job的剩余工作量
        self.res_edge_data_volume=-1                                     #该job的剩余数据流边的数据量
        self.res_num_edges=-1                                            #该job的剩余数据边数量
        self.res_stat_amount=-1                                          #剩余状态量的大小
        self.sign_job_finished=False                                     #标志其是否已部署完成
        self.lis_mds_finished=[]                                         #指示该job已调度执行的模块0-1列表（按完成时间的先后排序,0代表已完成的模块）

        self.indg_redep_mat=np.zeros((num_mds,num_mds,args.num_processors),dtype=np.float32)                          #记录该job入度module的冗余和部署情况的邻接矩阵（number of mds *number of mds，element：red & dep）需要保证为list列表类型

        self.start_time=-1                                            #该job的开始执行时间
        self.end_time=-1                                              #该job的结束执行时间
        self.make_span=-1                                             #该job的makespan

        self.lis_redun_pos=[]                                         #各个冗余模块的执行位置（总模块数*总processors数）
        self.mat_retransmission=[]                                    #表示重传次数的邻接矩阵，扩展的重传次数-邻接矩阵，即冗余-传输次数邻接矩阵
        self.reliability_real=-1                                      #实际达到的可靠性reliability

        self.lis_redundancy=[]                                        #各个module的冗余度列表
        self.lis_retrans=[]                                           #记录各个md的重传次数决策


    # def Dec_res_num_mds(self):
    #     self.res_num_mds-=1


    # def Upt_res_num_mds(self):
    #     res_num_mds=0
    #     res_md_op_amound=0
    #     for md in self.lis_mds:
    #         if md.sign_md_finished:
    #             res_num_mds+=1
    #             res_md_op_amound+=md.op_amount
    #     self.res_num_mds=res_num_mds
    #     self.res_md_op_amound=res_md_op_amound


    def Set_lis_red(self):
        for md in self.lis_mds:
            self.lis_redundancy.append(md.lis_processors)

    def Set_lis_retrans(self):
        for md in self.lis_mds:
            self.lis_retrans.append(md.retrans_act)

    # def SetMS(self):
    #     self.make_span=self.end_time-self.start_time

    def Set_Make_Span(self):
        self.make_span=self.end_time-self.release_time

    def Set_lis_mds(self,lis_mds):
        self.lis_mds=lis_mds

    # def Set_total_md_op_amount(self,total_md_op_amount):
    #     self.total_md_op_amount=total_md_op_amount

    def Set_release_time(self,release_time):
        self.release_time=release_time

    #更新job的入度邻接矩阵
    def Upt_indg_redep_mat(self,module):
        md_idx=self.lis_mds.index(module)
        # for i in range(self.num_mds):
        #     if self.ori_adj_mat[md_idx][i]==1:
        #         self.indg_redep_mat[md_idx][i]=module.lis_processors
        for sub_md in module.lis_sub_mods:
            self.indg_redep_mat[md_idx][self.lis_mds.index(sub_md)]=module.lis_processors

    #实现在一个module的冗余度确定之后，该job的冗余邻接矩阵的修改
    def Upt_mat_red(self,module):
        md_idx=self.lis_mds.index(module)
        self.lis_redundancy.append(module.redundancy)
        mat_idx=sum(self.lis_redundancy[:md_idx])
        arr_row_md=self.mat_retransmission[mat_idx]
        pre_mat=self.mat_retransmission[:mat_idx+1]
        sub_mat=self.mat_retransmission[mat_idx+1:]
        inc_rows_md=[arr_row_md for _ in range(module.redundancy-1)]
        self.mat_retransmission=np.vstack((pre_mat,inc_rows_md,sub_mat))
        mat_col=self.mat_retransmission.T
        arr_col_md=mat_col[mat_idx]
        pre_mat_1=mat_col[:mat_idx+1]
        sub_mat_1=mat_col[mat_idx+1:]
        inc_cols_md=[arr_col_md for _ in range(module.redundancy-1)]
        self.mat_retransmission=np.vstack(pre_mat_1,inc_cols_md,sub_mat_1).T

    def Get_num_pre_red_mds(self,module):
        num_pre_red_mds=0
        for md in module.lis_pre_mods:
            num_pre_red_mds+=md.redundancy
        return num_pre_red_mds

    def JudMd(self,module):
        if module in self.lis_mds:
            return True
        else:
            return False

    def GetIndxMd(self,moudle):
        return self.lis_mds.index(moudle)





