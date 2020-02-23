class module(object):
    def __init__(self,idx,job,op_amount,lis_pre_mods,lis_sub_mods,is_stateful,state,layer):
        self.idx=idx                                 #module id
        self.job=job                                 #所在的job对象
        self.op_amount=op_amount                     #需要的运算量（MB）
        if lis_pre_mods:
            self.lis_pre_mods=lis_pre_mods           #前驱模块对象列表
        else:
            self.lis_pre_mods=[]
        if lis_sub_mods:
            self.lis_sub_mods=lis_sub_mods           #后继模块对象列表
        else:
            self.lis_sub_mods=[]
        # self.lis_sub_mods=lis_sub_mods
        self.is_stateful=is_stateful                 #标识该module是否有状态
        self.state=state                             #如果为有状态的模块，其状态量对象
        self.layer=layer                             #所在层数
        self.dic_edges_sub_mds={}                    #与后继modules之间的数据流对象字典  ***

        self.sign_md_finished=False                  #标志该module是否已完成

        self.edge_cloud=None                         #被部署的edge cloud对象
        self.processor=None                          #被部署的processor对象（其包含了edge cloud信息）
        self.lis_processors=None                     #被部署的processors标志列表
        self.retrans_act=None                        #该md的重传次数矩阵列表

        self.release_time = -1                       #释放时间
        self.wait_time=-1                            #等待时间
        self.start_time=-1                           #开始执行时间
        self.execut_time=-1                          #执行时间
        self.residual_exe_time=-1                    #剩余执行时间（抢占时考虑）
        self.end_time=-1                             #结束执行时间
        self.redundancy=-1                           #冗余度
        self.lis_redund=[]                           #冗余模块对象列表
        self.indegree=0
        self.reliability=0                           #该module达到的可靠性

        self.entry_edge=None                         #如果该module为entry module，那么此为其接收的数据边

        self.exit_edge=None                          #如果该module为exit module，那么此为其最后输出的数据边

        self.earliest_start_time=-1                  #所有前驱red_md的最早结束时间

        # self.lis_red_release_time=[]

    def Set_earliest_start_time(self):
        earliest_start_time = 100000
        for pre_md in self.lis_pre_mods:
            for pre_re_md in pre_md.lis_redund:
                if pre_re_md.end_time < earliest_start_time:
                    earliest_start_time = pre_re_md.end_time
        self.earliest_start_time=earliest_start_time


    def Set_Entry_Edge(self,entry_edge):
        self.entry_edge=entry_edge

    def Set_Exit_Edge(self,exit_edge):
        self.exit_edge=exit_edge

    def Set_edge_cloud(self,edge_cloud):
        self.edge_cloud=edge_cloud

    def Set_job(self,job):
        self.job=job

    def Add_pre_mod(self,md):
        self.lis_pre_mods.append(md)

    def Add_sub_mod(self,md):
        self.lis_sub_mods.append(md)

    def Set_indegree(self,indegree):
        self.indegree=indegree

    def Sort_lis_sub_mods(self):
        self.lis_sub_mods=sorted(self.lis_sub_mods,key=lambda s:s.idx)

    def Set_redundancy(self,redundancy):
        self.redundancy=redundancy

    def Set_lis_processors(self,lis_processors):
        self.lis_processors=lis_processors