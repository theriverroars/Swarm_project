import asyncio
from threading import Thread
import numpy as np
import qtm
from collections import deque
import xml.etree.ElementTree as ET
import pkg_resources

class  QTMSDK(Thread):
    """Run QTM Wrapper on its own thread."""
    def __init__(self, qtm_ip):
        Thread.__init__(self)

        # self.body_name = body_name
        self.on_pose = None
        self.connection = None
        self._stay_open = True
        self.qtm_ip = qtm_ip
        self._markers = None
        self.body_index = None
        self.bodydict = {}

        self.start()

    def close(self):
        self._stay_open = False
        self.join()
    
    def run(self):
        asyncio.run(self._life_cycle())

    async def _life_cycle(self):
        await self._connect()
        while self._stay_open:
            await asyncio.sleep(1)
        await self._close()

    def create_body_index(self,xml_string):
        """ Extract a name to index dictionary from 6dof settings xml """
        xml = ET.fromstring(xml_string)

        body_to_index = {}
        for index, body in enumerate(xml.findall("*/Body/Name")):
            body_to_index[body.text.strip()] = index

        return body_to_index

    async def find_bodies(self):
        xml_string = await self.connection.get_parameters(parameters=["6d"])
        self.body_index = self.create_body_index(xml_string)

    async def _connect(self): 
        host = self.qtm_ip
        print('Connecting to QTM on ' + host)
        self.connection = await qtm.connect(host)
        await self.find_bodies()
        await self.connection.stream_frames(components=['6d'], on_packet=self._on_packet)
        
        # await asyncio.sleep(1000)
        # await self.connection.stream_frames_stop()

    def get_key(self,val,dict):
        for key,value in dict.items():
            if val==value:
                return key
        return "No such Body"
    
    def _on_packet(self, packet):
        info, bodies = packet.get_6d()
        i = 0
        # Print all bodies
        for position, rotation in bodies:
            body_name = self.get_key(i,self.body_index)
            # print("body : {} - Pos: {}".format(body_name,position))
            self.bodydict.update({body_name:[position.x,position.y,position.z,rotation]}) 
            i = i+1

        # self._current_frame = packet.framenumber
        # _, self._markers = packet.get_3d_markers()
        # X = []
        # Y = []
        # for i in self._markers:
        #     X.append(i.x)
        #     Y.append(i.y)    
        # self.current_x = np.average(np.array(X))
        # self.current_y = np.average(np.array(Y))

    def getxy(self):
        return self.bodydict

    async def _close(self):
        await self.connection.stream_frames_stop()
        self.connection.disconnect()