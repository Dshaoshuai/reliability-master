class red_edge(object):
    def __init__(self,idx,pre_red_md,sub_red_md,sour_edge,data_volume):
        self.idx=idx
        self.pre_red_md=pre_red_md
        self.sub_red_md=sub_red_md
        self.sour_edge=sour_edge
        self.data_volume=data_volume

        self.net_env=None
        self.net_channel=None
        self.release_time=-1
        self.wait_time=-1
        self.start_time=-1
        self.trans_time=-1               #传输时延
        self.ori_propg_time=-1           #原始传播时间，即一次传播的时间
        self.end_time=-1


        self.trans_suc=-1                #一次传输的成功率
        self.exp_num_trans=-1            #期望的传播次数（服从几何分布，用于计算makespan）
        self.exp_propg_time=-1           #期望的传播时间，在计算完成时间时，按照期望的传播时间来计算
        self.num_retrans=0               #重传次数
        self.reliability=-1              #该数据流边达到的可靠性
        self.reliability_pre=-1          #该数据流边和其前驱red_md共同的可靠性

    def Set_Release_time(self,release_time):
        self.release_time=release_time

    def Set_reliability(self):
        reliability=0
        for i in range(1,self.num_retrans+2):
            reliability+=pow(1-self.trans_suc,i-1)*self.trans_suc
        self.reliability=reliability

    def Set_reliability_pre(self,pre_re_md):
        self.reliability_pre=self.reliability*pre_re_md.reliability

    def Set_Start_time(self,start_time):
        self.start_time=start_time

    def Set_net_env(self,net_env):
        self.net_env=net_env

    def Set_net_channel(self,net_channel):
        self.net_channel=net_channel

    def Set_end_time(self):
        self.end_time=self.start_time+self.trans_time+self.exp_propg_time          #计算几何分布的期望值
