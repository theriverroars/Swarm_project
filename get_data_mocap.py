import time
import numpy as np
from mocapSDK import QTMSDK

QTM_IP = "192.168.0.2"

mocap = QTMSDK(QTM_IP)
time.sleep(1)

if __name__=="__main__":

    try:
        while 1:
            bodydict = mocap.getxy()
            body_names = list(bodydict.keys())
            xy_vals = list(bodydict.values())
            print("received new data")
            print(bodydict)
            # for i in range(len(bodydict)):
            #     rel_dist = relDist()
            #     rel_dist.objName = bytes(body_names[i],'utf-8')
            #     rel_dist.xrel = xy_vals[i][0]
            #     rel_dist.yrel = xy_vals[i][1]
            #     rel_dist_vec.relDistVec.append(rel_dist)
            # print(rel_dist_vec)
    except KeyboardInterrupt:
        print("closing mocap")
        mocap.close()