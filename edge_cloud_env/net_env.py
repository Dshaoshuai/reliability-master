import copy

class net_env(object):
    def __init__(self,idx,pre_ec,sub_ec,num_net_channels,is_heterog,lis_net_channels,err_prob_tran):
        self.idx=idx                                             #网络环境对象的id
        # if lis_adj_edge_clouds:
        #     self.lis_adj_edge_clouds=lis_adj_edge_clouds         #相邻的两个edge clouds对象列表
        # else:
        #     self.lis_adj_edge_clouds=[]
        self.pre_ec=pre_ec                                       #前驱edge cloud
        self.sub_ec=sub_ec                                       #后继edge cloud
        self.num_net_channels=num_net_channels                   #网络信道的数量
        self.is_heterog=is_heterog                               #是否为异构网络环境
        if lis_net_channels:
            self.lis_net_channels=lis_net_channels               #网络信道对象列表
        else:
            self.lis_net_channels=[]

        self.cur_edge = None                                      # 当前正在传输的数据边对象

        self.lis_finished_edges=[]                                #传输成功的数据边对象列表（历史）
        self.total_data_volume=0                                  #传输成功的总数据流量
        self.lis_wait_edges = []                                  # 当前正在该net的等待队列中的数据边对象列表
        self.total_data_volume_wait=0                             # 当前正在该net的等待队列中的数据边流量

        self.err_prob_tran=err_prob_tran                          #固有的传输出错概率，假设相同net_env中的网络信道是同构的

        self.earliest_avail_time=0                                #该net的最早可用时间
        self.lis_ear_channels=[]                                  #最早可用的信道列表，初始化为所有的信道，需要在运行时实时更新

    # def Get_adj_edge_clouds(self):
    #     return self.adj_edge_clouds

    #更新该net env的等待队列
    def Updat_lis_wait_edges(self,red_eg):
        for eg in self.lis_wait_edges:
            if red_eg.release_time>=eg.start_time and red_eg.release_time<eg.end_time:
                self.cur_edge=eg
                self.lis_wait_edges.remove(eg)
            elif eg.start_time<red_eg.release_time:
                self.lis_wait_edges.remove(eg)
        if red_eg.release_time<self.earliest_avail_time:
            self.lis_wait_edges.append(red_eg)
        else:
            self.cur_edge=red_eg

        total_dv_wait=0
        for eg in self.lis_wait_edges:
            total_dv_wait+=eg.data_volume
        self.total_data_volume_wait=total_dv_wait


    def Set_earliest_avail_time(self):
        earliest_avail_time=100000
        for channel in self.lis_net_channels:
            if earliest_avail_time>=channel.earliest_avail_time:
                earliest_avail_time=channel.earliest_avail_time
        self.earliest_avail_time=earliest_avail_time

    def Set_lis_ear_channels(self):
        # print('1_self.lis_ear_channels: ', self.lis_ear_channels)
        # print('1_self.lis_net_channels: ',self.lis_net_channels)
        self.lis_ear_channels.clear()
        # print('self.lis_ear_channels: ',self.lis_ear_channels)
        # print('self.lis_net_channels: ', self.lis_net_channels)
        # print('earliest_avail_time: ',self.earliest_avail_time)
        # print('lis channals earliest_avail_time: ',[channel.earliest_avail_time for channel in self.lis_net_channels] )
        for channel in self.lis_net_channels:
            if channel.earliest_avail_time==self.earliest_avail_time:
                self.lis_ear_channels.append(channel)

    def Set_lis_net_channels(self,lis_net_channels):
        self.lis_net_channels=lis_net_channels

    def Get_err_prob_tran(self):
        return self.err_prob_tran

    def reset(self):
        self.earliest_avail_time=0
        self.lis_ear_channels=copy.deepcopy(self.lis_net_channels)
        for channel in self.lis_net_channels:
            channel.reset()