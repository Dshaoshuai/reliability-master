class state(object):
    def __init__(self,sour_red_module,sour_ori_md,job,amount):
        self.sour_red_module=sour_red_module              #所在的冗余模块对象
        self.sour_ori_md=sour_ori_md                      #所在的原始module对象
        self.job=job                                      #所在的job对象
        self.amount=amount                                #状态量大小
        self.edge_cloud=None                              #所在的边缘云对象
        self.processor=None                               #所在的processor对象
        self.start_mig=-1                                 #开始迁移时间
        self.mig_time=-1                                  #迁移时间
        self.end_mig=-1                                   #迁移结束时间

    def Set_edge_cloud(self,edge_cloud):
        self.edge_cloud=edge_cloud

    def Set_sour_ori_md(self,md):
        self.sour_ori_md=md