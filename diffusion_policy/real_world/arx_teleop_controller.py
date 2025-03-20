import os
import time
import enum
import sys
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
from diffusion_policy.real_world.arx_client import Arx5Client
# from diffusion_policy.real_world.arx_interface import ARXInterface
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


class Command(enum.Enum):
    STOP = 0
    START = 1


class ARXTeleOpController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """


    def __init__(self,
            shm_manager: SharedMemoryManager, 
            robot_ip,
            master_port = 8765,
            slave_port = 8766,
            frequency=125, 
            lookahead_time=0.1, 
            gain=300,
            max_pos_speed=0.25, # 5% of max speed
            max_rot_speed=0.16, # 5% of max speed
            launch_timeout=3,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128,
            ):
        """
        frequency: CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose
        payload_mass: float
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.

        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="ARXPositionalController")
        self.robot_ip = robot_ip
        self.master_port = master_port
        self.slave_port = slave_port
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # build input queue
        example = {
            'cmd': Command.STOP.value
        }

        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        example = dict()

        example["actual_eef_pose"] = np.zeros(6)
        example["actual_joint_pos"] = np.zeros(6)
        example["actual_joint_vel"] = np.zeros(6)
        example["actual_joint_torque"] = np.zeros(6)
        example["actual_gripper_pos"] = 0.0
        example["actual_gripper_vel"] = 0.0
        example["actual_gripper_torque"] = 0.0
        # example["actual_gain"] = 0.0
        example['robot_receive_timestamp'] = time.time()

        master_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        slave_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.master_ring_buffer = master_ring_buffer
        self.slave_ring_buffer = slave_ring_buffer
        self.receive_keys = receive_keys
    
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[ARXPositionalController] Controller process spawned at {self.pid}")

        message = {
            'cmd': Command.START.value,
        }
        print(f"[ARXPositionalController] Reset to home")
        self.input_queue.put(message)

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value,
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= receive APIs =============
    def get_master_state(self, k=None, out=None):
        if k is None:
            return self.master_ring_buffer.get(out=out)
        else:
            return self.master_ring_buffer.get_last_k(k=k,out=out)
    
    def get_master_all_state(self):
        return self.master_ring_buffer.get_all()
    
    def get_slave_state(self, k=None, out=None):
        if k is None:
            return self.slave_ring_buffer.get(out=out)
        else:
            return self.slave_ring_buffer.get_last_k(k=k,out=out)
    
    def get_slave_all_state(self):
        return self.master_ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start rtde
        robot_ip = self.robot_ip

        arx_master = Arx5Client(robot_ip, self.master_port)
        arx_slave = Arx5Client(robot_ip, self.slave_port)

        try:
            if self.verbose:
                print(f"[ARXPositionalController] Connect to master arm: {self.master_port}")
                print(f"[ARXPositionalController] Connect to slave arm: {self.slave_port}")

            # set parameters
            # if self.tcp_offset_pose is not None:
            #     rtde_c.setTcp(self.tcp_offset_pose)
            # if self.payload_mass is not None:
            #     if self.payload_cog is not None:
            #         assert rtde_c.setPayload(self.payload_mass, self.payload_cog)
            #     else:
            #         assert rtde_c.setPayload(self.payload_mass)
            
            # # init pose
            # if self.joints_init is not None:
            #     assert rtde_c.moveJ(self.joints_init, self.joints_init_speed, 1.4)
            arx_master.reset_to_home()
            arx_slave.reset_to_home()       


            # # main loop
            # dt = 1. / self.frequency
            # # curr_pose = rtde_r.getActualTCPPose()
            # state_data = arx_robot.get_state()["data"]

            # # use monotonic time to make sure the control loop never go backward
            # curr_t = time.monotonic()
            # last_waypoint_time = curr_t
            # pose_interp = PoseTrajectoryInterpolator(
            #     times=[curr_t],
            #     poses=[state_data['ee_pose']]
            # )

            # Set Master Arm free to move
            arx_master.set_to_damping()
            gain = arx_master.get_gain()
            gain["kd"] = gain["kd"] * 0.1
            gain["gripper_kd"] = gain["gripper_kd"] * 0.1
            arx_master.set_gain(gain)
            
            iter_idx = 0
            keep_running = True
            following = False
            while keep_running:
                # start control iteration
                # t_start = rtde_c.initPeriod()
                # t_start = time.perf_counter()

                # send command to robot
                t_start = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                # pose_command = pose_interp(t_now)
                # vel = 0.5
                # acc = 0.5
                # assert rtde_c.servoL(pose_command, 
                #     vel, acc, # dummy, not used by ur5
                #     dt, 
                #     self.lookahead_time, 
                #     self.gain)
                # arx_slave.set_ee_pose(pose_command[:6],pose_command[6])
                
                # update robot state
                state = dict()
                # for key in self.receive_keys:
                #     state[key] = np.array(getattr(rtde_r, 'get'+key)())
                state_data = arx_slave.get_state()
                state["actual_eef_pose"] = state_data["ee_pose"]
                state["actual_joint_pos"] = state_data["joint_pos"]   
                state["actual_joint_vel"] = state_data["joint_vel"]   
                state["actual_joint_torque"] = state_data["joint_torque"] 
                state["actual_gripper_pos"] = np.array(state_data["gripper_pos"]).reshape(1)
                state["actual_gripper_vel"] = np.array(state_data["gripper_vel"]).reshape(1)
                state["actual_gripper_torque"] = np.array(state_data["gripper_torque"]).reshape(1)
                state['robot_receive_timestamp'] = time.time()
                self.slave_ring_buffer.put(state)

                state_data = arx_master.get_state()
                state["actual_eef_pose"] = state_data["ee_pose"]
                state["actual_joint_pos"] = state_data["joint_pos"]   
                state["actual_joint_vel"] = state_data["joint_vel"]   
                state["actual_joint_torque"] = state_data["joint_torque"] 
                state["actual_gripper_pos"] = np.array(state_data["gripper_pos"]).reshape(1)
                state["actual_gripper_vel"] = np.array(state_data["gripper_vel"]).reshape(1)
                state["actual_gripper_torque"] = np.array(state_data["gripper_torque"]).reshape(1)
                state['robot_receive_timestamp'] = time.time()
                self.master_ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                if n_cmd > 0:
                    cmd = commands['cmd'][0]    # Only get the frist command
                    
                    if cmd == Command.STOP.value:
                        following = False
                    else:
                        following = True

                # Following
                if following:
                    master_state = arx_master.get_state()
                    # gain = arx_slave.get_gain()
                    state = dict()
                    state["ee_pose"] = master_state["ee_pose"]
                    state["joint_pos"] = master_state["joint_pos"]   
                    state["joint_vel"] = master_state["joint_vel"]   
                    state["joint_torque"] = master_state["joint_torque"] 
                    state["gripper_pos"] = master_state["gripper_pos"] * 4.0
                    state["gripper_vel"] = master_state["gripper_vel"]
                    state["gripper_torque"] = master_state["gripper_torque"]
                    state['robot_receive_timestamp'] = time.time()
                    arx_slave.set_ee_pose(state["ee_pose"], state["gripper_pos"])


                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                
                time.sleep(1/self.frequency - (time.monotonic() - t_start))
                # print(f"[ARXPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

                if self.verbose:
                    print(f"[ARXPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            # manditory cleanup
            # decelerate
            # rtde_c.servoStop()

            # # terminate
            # rtde_c.stopScript()
            # rtde_c.disconnect()
            # rtde_r.disconnect()
            arx_master.reset_to_home()
            arx_slave.reset_to_home()
            self.ready_event.set()

            if self.verbose:
                print(f"[ARXPositionalController] Disconnected from master arm: {master_ip} and slave arm: {slave_ip}")
