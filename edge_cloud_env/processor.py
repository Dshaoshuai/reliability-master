class processor(object):
    def __init__(self,idx,edge_cloud,cap_process,is_idle,err_prob_op):
        self.idx=idx                                  #processor id
        self.edge_cloud=edge_cloud                    #所在的edge cloud对象
        self.cap_process=cap_process                  #处理能力（MB/s）
        self.num_sum_modules=-1                       #冗余模块的数量
        self.lis_sum_modules=[]                       #冗余模块的对象列表
        self.cur_exe_module=None                      #当前执行的module对象
        self.lis_wait_modules=[]                      #当前在等待的module对象列表
        self.total_op_amount_wait=0                   #等待队列中的总运算量
        self.is_idle=is_idle                          #标志是否为空闲（True/False），初始化都设为True
        self.earlist_aval_time=-1                     #最早可用时间
        self.err_prob_op=err_prob_op                  #该processor的固有执行出错概率

        self.lis_edges=[]                             #该processor迄今为止输出的总数据边列表
        self.total_data_volume=0                      #该processor迄今为止输出的总数据流量
        self.lis_edges_wait=[]                        #当前该processor输出的edges中等待传出还没有传输的edge列表
        self.total_data_volume_wait=0                 #当前该processor输出的edges中等待传输的总数据流量


        self.lis_stat_mds=[]                          #该processor上有状态的模块列表
        self.total_stat=0                             #该processor上的状态量总量

    def Set_cur_exe_module(self,cur_exe_module):
        self.cur_exe_module=cur_exe_module

    def Updat_lis_wait_modules(self,md):
        for mod in self.lis_wait_modules:
            if mod.start_time < md.release_time and mod.end_time > md.release_time:
                self.cur_exe_module = mod
                self.lis_wait_modules.remove(mod)
            elif mod.start_time < md.release_time:
                self.lis_wait_modules.remove(mod)
        if md.release_time<self.earlist_aval_time:
            self.lis_wait_modules.append(md)
        else:
            self.cur_exe_module=md

        total_dv_wait=0
        for mod in self.lis_wait_modules:
            total_dv_wait+=mod.op_amount
        self.total_data_volume_wait=total_dv_wait



    def Set_earlist_aval_time(self,earliest_aval_time):
        self.earlist_aval_time=earliest_aval_time

    def reset(self):
        self.num_sum_modules=0
        self.lis_sum_modules.clear()
        self.cur_exe_module=None
        self.lis_wait_modules.clear()
        self.is_idle=True
        self.earlist_aval_time=0
