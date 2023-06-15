import math
import asyncio
from threading import Thread
import qtm
import xml.etree.cElementTree as ET


def sqrt(x):
    """Calculate sqrt while avoiding rounding errors with slightly negative x."""
    if x < 0.0:
        return 0.0
    return math.sqrt(x)

class Pose:
    """Holds pose data with euler angles and/or rotation matrix"""
    def __init__(self, x, y, z, roll=None, pitch=None, yaw=None, rotmatrix=None):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.rotmatrix = rotmatrix

    @classmethod
    def from_qtm_6d(cls, qtm_6d):
        """Build pose from rigid body data in QTM 6d component"""
        qtm_rot = qtm_6d[1].matrix
        cf_rot = [[qtm_rot[0], qtm_rot[3], qtm_rot[6]],
                  [qtm_rot[1], qtm_rot[4], qtm_rot[7]],
                  [qtm_rot[2], qtm_rot[5], qtm_rot[8]]]
        return cls(qtm_6d[0][0] / 1000,
                   qtm_6d[0][1] / 1000,
                   qtm_6d[0][2] / 1000,
                   rotmatrix = cf_rot)

    @classmethod
    def from_qtm_6deuler(cls, qtm_6deuler):
        """Build pose from rigid body data in QTM 6deuler component"""
        return cls(qtm_6deuler[0][0] / 1000,
                   qtm_6deuler[0][1] / 1000,
                   qtm_6deuler[0][2] / 1000,
                   roll  = qtm_6deuler[1][2],
                   pitch = qtm_6deuler[1][1],
                   yaw   = qtm_6deuler[1][0])

    def distance_to(self, other_point):
        return sqrt(
            (self.x - other_point.x) ** 2 +
            (self.y - other_point.y) ** 2 +
            (self.z - other_point.z) ** 2)

    def is_valid(self):
        """Check if any of the coodinates are NaN."""
        return self.x == self.x and self.y == self.y and self.z == self.z

    def __str__(self):
        return "x: {:6.2f} y: {:6.2f} z: {:6.2f} Roll: {:6.2f} Pitch: {:6.2f} Yaw: {:6.2f}".format(
            self.x, self.y, self.z, self.roll, self.pitch, self.yaw)

class QtmWrapper(Thread):
    """Run QTM connection on its own thread."""
    def __init__(self, qtm_ip, cf_body_name):
        Thread.__init__(self)

        self.on_cf_pose = None
        self.connection = None
        self.bodyToIdx = {}
        self._stay_open = True

        self.qtm_ip = qtm_ip
        self.cf_body_name = cf_body_name

        self.cf_trackingLoss = 0

        self.start()

    def close(self):
        self._stay_open = False
        self.join()

    def run(self):
        asyncio.run(self._life_cycle())

    async def _life_cycle(self): 
        await self._connect()
        while(self._stay_open):
            await asyncio.sleep(1)
        await self._close()

    async def _connect(self):
        print('Connecting to QTM at ' + self.qtm_ip)
        self.connection = await qtm.connect(self.qtm_ip)

        params_xml = await self.connection.get_parameters(parameters=['6d'])
        xml = ET.fromstring(params_xml)
        for index, body in enumerate(xml.findall("*/Body/Name")):
            self.bodyToIdx[body.text.strip()] = index
        print('QTM 6DOF bodies and indexes: ' + str(self.bodyToIdx))

        # Check if all the bodies are there

        if self.cf_body_name in self.bodyToIdx:
            print("Crazyflie body '" + self.cf_body_name + "' found in QTM 6DOF bodies.")
        else:
            print("Crazyflie body '" + self.cf_body_name + "' not found in QTM 6DOF bodies!")
            print("Aborting...")
            self._stay_open = False

        # for controller_body_name in controller_body_names:
        #     if controller_body_name in self.bodyToIdx:
        #         print("Controller body '" + controller_body_name + "' found in QTM 6DOF bodies.")
        #     else:
        #         print("Controller body '" + controller_body_name + "' not found in QTM 6DOF bodies!")
        #         print("Aborting...")
        #         self._stay_open = False

        await self.connection.stream_frames(components=['6d', '6deuler'], on_packet=self._on_packet)


    def _on_packet(self, packet):
        global cf_pose, controller_poses
        # We need the 6d component to send full pose to Crazyflie,
        # and the 6deuler component for convenient calculations
        header, component_6d = packet.get_6d()
        header, component_6deuler = packet.get_6d_euler()

        if component_6d is None:
            print('No 6d component in QTM packet!')
            return              
        
        if component_6deuler is None:
            print('No 6deuler component in QTM packet!')
            return      

        # Get 6DOF data for Crazyflie
        cf_6d = component_6d[self.bodyToIdx[self.cf_body_name]]
        # Store in temp until validity is checked
        _cf_pose = Pose.from_qtm_6d(cf_6d)
        # Check validity
        if _cf_pose.is_valid():
            # Update global var for pose
            cf_pose = _cf_pose
            #print(f'mocap pos: x: {cf_pose.x}, y: {cf_pose.y}, z: {cf_pose.z}')
            # Stream full pose to Crazyflie
            if self.on_cf_pose:
                self.on_cf_pose([cf_pose.x, cf_pose.y, cf_pose.z, cf_pose.rotmatrix])
                self.cf_trackingLoss = 0
        else:
            self.cf_trackingLoss += 1
    
    def getpose(self):
        global cf_pose
        return cf_pose

        # Get 6DOF data for controllers and update globals

        # for i, controller_body_name in enumerate(controller_body_names):
        #     controller_6deuler = component_6deuler[self.bodyToIdx[controller_body_name]]
        #     _controller_pose = Pose.from_qtm_6deuler(controller_6deuler)
        #     if _controller_pose.is_valid():
        #         controller_poses[i] = _controller_pose

    async def _close(self):
        await self.connection.stream_frames_stop()
        self.connection.disconnect()