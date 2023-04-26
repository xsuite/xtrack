
class PipelineStatus:
    def __init__(self, on_hold, data=None, info=None):
        self.on_hold = on_hold
        self.data = data
        self.info = info

class PipelineID:
    def __init__(self,rank,number=0):
        self.number = number
        self.rank = rank

class PipelineCommunicator:

    def __init__(self):
        self.messages = {}

    def Issend(self,send_buffer,dest,tag):
        if dest not in self.messages.keys():
            self.messages[dest] = {}
        if tag not in self.messages[dest].keys():
            self.messages[dest][tag] = []
        self.messages[dest][tag].append(send_buffer)
        return self

    def Recv(self,recieve_buffer,source,tag):
        assert source in self.messages.keys()
        assert tag in self.messages[source].keys()
        assert bool(self.messages[source][tag])
        message = self.messages[source][tag].pop(0)
        recieve_buffer[:] = message[:]

    def Iprobe(self,source, tag):
        if source in self.messages.keys():
            if tag in self.messages[source].keys():
                return bool(self.messages[source][tag])
        return False

    def Test(self):
        return True
