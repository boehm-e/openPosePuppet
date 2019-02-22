import bpy
import math
import numpy as np

human = bpy.data.objects['free3dmodel_skeleton']
bpy.context.scene.objects.active = human
bpy.ops.object.mode_set(mode='POSE')

bone = human.pose.bones["shoulder.L"]
import time



def SetBoneRotationDeg(human,boneName,rotEulerDeg):
    lastMode = human.pose.bones[boneName].rotation_mode
    human.pose.bones[boneName].rotation_mode = 'XYZ'    
    human.pose.bones[boneName].rotation_euler[0] = math.radians(rotEulerDeg[0])
    human.pose.bones[boneName].rotation_euler[1] = math.radians(rotEulerDeg[1])
    human.pose.bones[boneName].rotation_euler[2] = math.radians(rotEulerDeg[2])
    human.pose.bones[boneName].rotation_mode = lastMode





import time, bpy, threading
def thread_update():
    while(True):
        with open('/tmp/coords.txt') as f:
            lines = f.readlines()
            print(lines)
            if (len(lines) > 1):
                leftRotate = float(lines[0].strip()) - 180
                SetBoneRotationDeg(human, "shoulder.L", [leftRotate,0,0] )

                rightRotate = float(lines[1].strip()) - 180
                SetBoneRotationDeg(human, "shoulder.R", [rightRotate,0,0] )
        time.sleep(0.1) #update rate in seconds

thread = threading.Thread(target=thread_update)
thread.start()
