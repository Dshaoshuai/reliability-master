import math

class red_module(object):
    def __init__(self,idx,source_mod,job,lis_pro_red_mods,is_stateful,state,op_amount):
        self.idx=idx                                           #冗余模块id
        self.source_mod=source_mod                             #源module对象
        self.job=job                                           #所在job对象
        if lis_pro_red_mods:
            self.lis_pro_red_mods=lis_pro_red_mods                 #前驱冗余模块列表
        else:
            self.lis_pro_red_mods=[]
        self.is_stateful=is_stateful                           #标识是否有状态（1：有状态，0：无状态）
        self.state=state                                       #如果有状态，其为状态量对象
        self.op_amount=op_amount


        self.edge_cloud=None                                   #被部署执行的edge cloud对象
        self.processor=None                                    #所在的processor对象
        self.arr_retrans=[]                                    #与前驱冗余modules集合之间的数据流重传次数列表（1*前驱modules的数量）
        self.lis_sub_red_mods=[]                               #后继冗余modules列表
        self.release_time=-1                                   #释放时间
        self.start_time=-1                                     #开始执行时间
        self.wait_time=-1                                      #等待时间
        self.exec_time=-1                                      #执行时间
        self.residual_exe_time=-1                              #剩余执行时间
        self.end_time=-1                                       #结束执行时间
        self.exe_suc=0                                         #执行成功的概率（只和自身的执行成功与否有关）
        self.reliability=0                                     #可靠性，关于所有前驱cross edges的传输可靠性和其自身的执行成功概率，其中前驱cross edge的传输可靠性包含其前驱md的可靠性
        # self.lis_processors=[]                               #

    def Set_edge_cloud(self,edge_cloud):
        self.edge_cloud=edge_cloud

    def Set_end_time(self):
        self.end_time=self.start_time+self.exec_time

    def Set_exe_suc(self,lamb):
        self.exe_suc=math.exp(-lamb*self.exec_time)