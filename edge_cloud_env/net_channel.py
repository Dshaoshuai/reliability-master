class net_channel(object):
    def __init__(self,idx,net_env,net_bandwidth,err_prob_tran):
        self.idx=idx                               #信道id
        self.net_env=net_env                       #所在的网络环境对象
        self.net_bandwidth=net_bandwidth           #具有的网络带宽

        self.lis_finished_edges=[]                 #传输成功的数据边对象列表（历史）
        self.cur_edge=None                         #当前正在传输的数据边对象
        self.lis_wait_edges=[]                     #当前正在该信道的等待队列中的数据边对象列表
        self.earliest_avail_time=0                 #最早可用时间（涉及排队？）
        self.err_prob_tran=err_prob_tran           #固有的传输出错概率

    def Set_cur_edge(self,cur_edge):
        self.cur_edge=cur_edge

    def Set_earliest_avail_time(self,earliest_avail_time):
        self.earliest_avail_time=earliest_avail_time

    def reset(self):
        self.lis_finished_edges.clear()
        self.cur_edge=None
        self.lis_wait_edges.clear()
        self.earliest_avail_time=0