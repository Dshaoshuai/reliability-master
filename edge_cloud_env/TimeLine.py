class TimeLine(object):
    def __init__(self):
        self.pq=[]
        # self.length=0

    def __len__(self):
        return len(self.pq)

    def peek(self):
        if len(self.pq)>0:
            return self.pq[0]
        else:
            return None

    def push(self,job):
        self.pq.append(job)

    def pop(self):
        if len(self.pq)>0:
            return self.pq.pop(0)
        else:
            return None

    def sort(self):                                           #将作业流中的job按释放时间先后排序
        return sorted(self.pq,key=lambda s:s.release_time)

    def reset(self):
        self.pq=[]