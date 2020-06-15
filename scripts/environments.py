import numpy as np
import socket
import struct
import time

class CartpoleEnv:
    def __init__(self, port_self, port_remote):
        print("Init CartpoleEnv")
        
        self.port_self_ = port_self
        self.port_remote_ = port_remote

        self.fd_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #self.fd_.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 112)       # set receive buffer size to be 112 bytes (1 double = 8 bytes)
        #self.fd_.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 40)        # set send buffer size to be 40 bytes. Both 112 bytes and 40 bytes are lower than the minimum value, so buffer size will be set as minimum value. See "'man 7 socket"
        self.fd_.bind( ("", self.port_self_) )
        self.addr_remote_send_ = ("", self.port_remote_)
        
        self.msg_len_ = 30
        self.data = [float(0)]*self.msg_len_

    def __del__(self):
        self.fd_.close() 

    def step(self):
        self.data[1] = 21
        buf = struct.pack('30f', *self.data)
        self.fd_.sendto(buf, self.addr_remote_send_)

        data, addr_remote_ = self.fd_.recvfrom(4*self.msg_len_)     # 1 float = 4 bytes, recvfrom is blocking
        unpacked_data = struct.unpack('30f',data)
        model_id,data_type,time = unpacked_data[0:3]
        position = unpacked_data[3]
        angle = unpacked_data[4]
        vel = unpacked_data[5]
        omega = unpacked_data[6]
        self.state_current_ = np.hstack([model_id,data_type,time,position,angle,vel,omega])
        state = np.asmatrix( self.state_current_[3:].reshape((-1,1)) )

        return state

    def reset(self, model_id:int):
        self.sendCommands(model_id=model_id, cmd_type="reset")
        time.sleep(0.5)
        state = self.step()

        return state
             
    def getTime(self):
        return self.state_current_[2]

    def sendCommands(self, model_id: int, cmd_type: str, command=0.0):
        self.data[0] = model_id
        if cmd_type=="force_cmd":
            self.data[1] = 10
            self.data[3] = command
        elif cmd_type == "reset":
            self.data[1] = 22
        else:
            print("undefined command type")
            return
        
        buf = struct.pack('30f', *self.data)
        self.fd_.sendto(buf, self.addr_remote_send_)




