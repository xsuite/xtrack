from xtrack.pipeline.core import PipelineID, PipelineCommunicator

class PipelineManager:
    def __init__(self,communicator = None):
        self._IDs = {}
        self._particles_per_rank = {}
        self._elements = {}
        self._pending_requests = {}
        self._last_request_turn = {}

        self.verbose = False

        if communicator is not None:
            self._communicator = communicator
        else:
            self._communicator = PipelineCommunicator()

    def add_particles(self,particles_name,rank):
        if rank in self._particles_per_rank.keys():
            pipeline_number = len(self._particles_per_rank[rank])
        else:
            pipeline_number = 0
            self._particles_per_rank[rank] = []
        pipeline_ID = PipelineID(rank,pipeline_number)
        self._IDs[particles_name] = pipeline_ID
        self._particles_per_rank[rank].append(particles_name)

    def get_particles_ID(self,particles_name):
        return self._IDs[particles_name]

    def get_particles_rank(self,particles_name):
        return self._IDs[particles_name].rank

    def add_element(self,element_name):
        self._elements[element_name] = len(self._elements)

    #
    # The tag is a int that identifies messages given the rank of the sender and of the reciever
    #
    def get_message_tag(self,element_name,sender_name,reciever_name,internal_tag=0):
        tag = self._elements[element_name] + len(self._elements)*self._IDs[sender_name].number + len(self._elements)*len(self._IDs)*self._IDs[reciever_name].number + len(self._elements)*len(self._IDs)*len(self._IDs)*internal_tag
        #if tag > self._max_tag:
        #    print(f'PyPLINEDElement WARNING {self.name}: MPI message tag {tag} is larger than max ({self._max_tag})')
        return tag

    #
    # The key is string that uniquely identifies a message
    #
    def get_message_key(self,element_name,sender_name,reciever_name,tag=0):
        return f'{element_name}_{sender_name}_{reciever_name}_{tag}'

    def is_ready_to_send(self,element_name,sender_name,reciever_name,turn,internal_tag=0):
        tag = self.get_message_tag(element_name=element_name,sender_name=sender_name,reciever_name=reciever_name,internal_tag=internal_tag)
        key = self.get_message_key(element_name=element_name,sender_name=sender_name,reciever_name=reciever_name,tag=tag)
        if key not in self._last_request_turn.keys():
            return True
        if turn <= self._last_request_turn[key]:
            if self.verbose:
                print(f'Pipeline manager {element_name}: {sender_name} at rank {self.get_particles_rank(sender_name)} has already sent a message to {reciever_name} at rank {self.get_particles_rank(reciever_name)} at turn {turn} with tag {tag}')
            return False
        if not self._pending_requests[key].Test():
            if self.verbose:
                print(f'Pipeline manager {element_name}: {sender_name} at rank {self.get_particles_rank(sender_name)} previous message to {reciever_name} at rank {self.get_particles_rank(reciever_name)} with tag {tag} was not receviced yet')
            return False
        return True

    def send_message(self,send_buffer,element_name,sender_name,reciever_name,turn,internal_tag=0):
        tag = self.get_message_tag(element_name=element_name,sender_name=sender_name,reciever_name=reciever_name,internal_tag=internal_tag)
        key = self.get_message_key(element_name=element_name,sender_name=sender_name,reciever_name=reciever_name,tag=tag)
        self._last_request_turn[key] = turn
        if self.verbose:
            print(f'Pipeline manager {element_name}: {sender_name} at rank {self.get_particles_rank(sender_name)} sending message to {reciever_name} at rank {self.get_particles_rank(reciever_name)} at turn {turn} with tag {tag}')
        self._pending_requests[key] = self._communicator.Issend(send_buffer,dest=self.get_particles_rank(reciever_name),tag=tag)

    def is_ready_to_recieve(self,element_name,sender_name,reciever_name,internal_tag=0):
        tag = self.get_message_tag(element_name=element_name,sender_name=sender_name,reciever_name=reciever_name,internal_tag=internal_tag)
        is_ready = self._communicator.Iprobe(source=self.get_particles_rank(sender_name), tag=tag)
        if self.verbose and not is_ready:
            print(f'Pipeline manager {element_name}: {reciever_name} at rank {self.get_particles_rank(reciever_name)} is not ready to recieve from {sender_name} at rank {self.get_particles_rank(sender_name)} with tag {tag}')
        return is_ready

    def recieve_message(self,recieve_buffer,element_name,sender_name,reciever_name,internal_tag=0):
        tag = self.get_message_tag(element_name=element_name,sender_name=sender_name,reciever_name=reciever_name,internal_tag=internal_tag)
        if self.verbose:
            print(f'Pipeline manager {element_name}: {reciever_name} at rank {self.get_particles_rank(reciever_name)} recieving from {sender_name} at rank {self.get_particles_rank(sender_name)} with tag {tag}')
        self._communicator.Recv(recieve_buffer,source=self.get_particles_rank(sender_name),tag=tag)
