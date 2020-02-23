class environment(object):
    def __init__(self,num_edge_clouds,lis_edge_clouds,dis_adj_mat,net_env_adj_mat,err_adj_mat,num_grades,lis_dist_ec_grade):
        self.num_edge_clouds=num_edge_clouds         #环境中的edge clous数量，包含移动设备（坐标位置（0，0））
        self.num_total_processors=-1
        self.lis_processors=[]
        if lis_edge_clouds:
            self.lis_edge_clouds=lis_edge_clouds         #edge clous对象列表
        else:
            self.lis_edge_clouds=[]
        if dis_adj_mat:
            self.dis_adj_mat=dis_adj_mat                 #edge clouds之间的距离-邻接矩阵
        else:
            self.dis_adj_mat=None
        if net_env_adj_mat:
            self.net_env_adj_mat=net_env_adj_mat         #edge clouds之间的网络环境-邻接矩阵，上三角矩阵 ###统一将上三角矩阵补全 （用!）已改
        else:
            self.net_env_adj_mat=None
        if err_adj_mat:
            self.err_adj_mat=err_adj_mat                 #edge clouds之间的网络的固有传输出错概率的邻接矩阵，上三角矩阵
        else:
            self.err_adj_mat=None

        self.lis_err_prob_op=None                        #edge clous固有执行出错概率的列表

        self.num_grades=num_grades                       #该执行环境中层级的数量
        self.lis_dist_ec_grade=lis_dist_ec_grade         #各个层级距离移动设备（坐标0，0）的距离（半径r）
        # self.dic_grade_class_ecs=dic_grade_class_ecs   #记录各个层级（grade）中不同类型（class）的edge cloud的数量列表的字典

        # self.wall_time=0

    def Get_dis_adj_mat(self):
        return self.dis_adj_mat

    def Set_lis_ecs(self,lis_ecs):
        self.lis_edge_clouds=lis_ecs

    def Set_dis_adj_mat(self,dis_adj_mat):
        self.dis_adj_mat=dis_adj_mat

    def Set_net_env_adj_mat(self,net_env_adj_mat):
        self.net_env_adj_mat=net_env_adj_mat

    def Set_err_adj_mat(self,err_adj_mat):
        self.err_adj_mat=err_adj_mat

    def Set_lis_err_prob_op(self,lis_err_prob_op):
        self.lis_err_prob_op=lis_err_prob_op

    def reset(self):
        for ec in self.lis_edge_clouds:
            self.num_total_processors+=ec.num_processors
            self.lis_processors.extend(ec.lis_processors)
        for ec in self.lis_edge_clouds:
            ec.reset()
        for i in range(self.num_edge_clouds):
            for j in range(self.num_edge_clouds):
                if i!=j:
                    self.net_env_adj_mat[i][j].reset()
