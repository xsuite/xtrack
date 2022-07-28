class PipelineStatus:
    def __init__(self, on_hold, data=None, info=None):
        self.on_hold = on_hold
        self.data = data
        self.info = info

class PipelineID:
    def __init__(self,rank,number=0):
        self.number = number
        self.rank = rank

