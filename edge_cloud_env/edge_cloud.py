import math

class edge_cloud(object):
    def __init__(self,idx,num_processors,lis_processors,adj_dic_net_env,avg_err_prob,grade):
        self.idx=idx                                      #边缘云id
        self.num_processors=num_processors                #processors的数量
        if lis_processors:
            self.lis_processors=lis_processors            #processors对象的列表
        else:
            self.lis_processors=[]
        # if adj_lis_edge_clouds:
        #     self.adj_lis_edge_clouds=adj_lis_edge_clouds   #相邻的边缘云对象列表（与所有的边缘云都相邻，故多余）
        # else:
        #     self.adj_lis_edge_clouds=[]
        if adj_dic_net_env:
            self.adj_dic_net_env=adj_dic_net_env          #到与其相邻的边缘云之间的网络环境字典
        else:
            self.adj_dic_net_env={}

        self.x_idx=-1                                     #该edge cloud位置的x坐标
        self.y_idx=-1                                     #该edge cloud位置的y坐标

        self.grade=grade                                  #所在的层级

        self.lis_idle_processors=[]                       #当前空闲的processors列表（按idx顺序排序后）

        self.lis_stat_mds=[]                              #该processor上有状态的模块列表
        self.total_stat=0                                 #该processor上的状态量总量

        self.cap_process=-1                               #该边缘云的处理能力（每个processor的处理能力*processors的数量【相同边缘云下不同processor处理能力同构】）
        self.avg_err_prob=avg_err_prob                    #该边缘云的平均出错概率（即为单个processor的出错概率【相同边缘云下不同processor出错概率同构】）
        self.num_red_modules=-1                           #在其上部署运行的（冗余）模块的数量（含已完成，即总体）
        self.lis_red_modules=[]                           #在其上部署运行的（冗余）模块的列表（含已完成，即总体）
        self.lis_cur_modules=[]                           #当前正在调度执行的模块列表（所有processors的正在调度模块总体）
        self.lis_cur_wai_modules=[]                       #当前正在等待队列中的模块列表（统计功能，包含其上所有processors的等待modules）

        self.earlist_avai_time=0                          #最早可用时间，表示为其上所有processors的最早可用时间的最小值（实现刷新功能）
        self.lis_mos_sui_processors=[]                    #当前最合适的processors列表，即为最小可用时间最小的processors列表（实现刷新功能） 注：处理器的调度粒度为processor，故在pg中用不到，但是在启发式方法中可能会用到

        # self.timer=0                                      #该edge cloud的计时器，每次用全局计时器更新，全局计时器每一轮采用

    def Set_cap_process(self,cap_process):
        self.cap_process=cap_process

    def Set_coord(self,pos_ec):
        self.x_idx=pos_ec[0]
        self.y_idx=pos_ec[1]

    def Set_lis_processors(self,lis_processors):
        self.lis_processors=lis_processors

    def Cal_dis_with_ec(self,ec):
        a=(self.x_idx,self.y_idx)
        b=(ec.x_idx,ec.y_idx)
        return round(math.sqrt(pow((a[0]-b[0]),2)+pow((a[1]-b[1]),2)),2)

    def reset(self):
        self.num_red_modules=0
        self.lis_red_modules.clear()
        self.lis_cur_modules.clear()
        self.lis_cur_wai_modules.clear()
        self.earlist_avai_time=0
        self.lis_idle_processors=self.lis_processors
        for processor in self.lis_processors:
            processor.reset()