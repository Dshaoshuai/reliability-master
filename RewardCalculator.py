class RewardCalculator(object):
    def __init__(self):
        # pass
        self.lis_finished_jobs=[]         #记录迄今为止执行结束的作业，计算reward时不仅要考虑当前调度的module带来的reward，还要考虑所有已完成jobs带来的reward

    def get_reward(self):
        pass

    def reset(self):
        # pass
        self.lis_finished_jobs.clear()