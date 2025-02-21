import os
import sys

curr_dir = os.getcwd()
ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "arx5-sdk")
ROOT_DIR = os.path.join(ROOT_DIR, "python")
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
from arx5_interface import Arx5CartesianController, EEFState, Gain
os.chdir(curr_dir)

class ARXInterface():
    def __init__(self, urdf_path = None, interface = None, name = "ARX Arm"):
        self.name = name
        if urdf_path is not None:
            urdf_path = os.path.join(ROOT_DIR, "../models/arx5.urdf")
        if interface is not None:
            interface = "can0"
        self.controller = Arx5CartesianController("L5", interface, urdf_path)
        self.robot_config = self.controller.get_robot_config()
        self.controller_config = self.controller.get_controller_config()

    def start(self):
        self.controller.reset_to_home()
        print(self.name + " Reset to home complete. Ready to start.")

    def getActualPose(self):
        eef_state = self.controller.get_eef_state()
        pass

    def getActualJointAngles(self):
        pass    

    def getActualJointVelocities(self):
        pass

    def getAcutalJointCurrent(self):
        pass

    def getTargetPose(self):
        pass    

    def __del__(self):
        self.controller.reset_to_home()
        print("Reset to home.")

        
