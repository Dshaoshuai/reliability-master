class edge(object):       #冗余模块之间的数据边
    def __init__(self,idx,job,pre_sour_md,sub_sour_md,pro_red_module,sub_red_module,is_cross_edge,data_volume):
        self.idx=idx                        #数据边id
        self.job=job                        #所在job对象
        self.pre_sour_md=pre_sour_md        #前驱源module对象
        self.sub_sour_md=sub_sour_md        #后继源module对象
        self.pro_red_module=pro_red_module  #前驱冗余模块对象
        self.sub_red_module=sub_red_module  #后继冗余模块对象
        self.is_cross_edge=is_cross_edge    #是否为交叉边
        self.data_volume=data_volume        #需要传输的数据量

        self.sign_edge_finished=False       #标识该edge是否传输完成

        self.num_retrans=-1                 #重传次数
        self.release_time=-1                #释放时间
        self.wait_time=-1                   #等待时间
        self.start_time=-1                  #开始传输时间
        self.trans_time=-1                  #传输时间
        self.end_time=-1                    #传输结束时间
        self.net_env=None                   #所在的网络环境对象
        self.net_channel=None               #所在的网络信道对象（其标识了其网络环境对象）

    def Set_num_retrans(self,num_retrans):
        self.num_retrans=num_retrans

    def Set_Release_time(self,release_time):
        self.release_time=release_time