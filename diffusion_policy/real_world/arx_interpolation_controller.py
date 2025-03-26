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
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    SET_GRIPPER = 3
    RESET_TO_HOME = 4
    SET_TO_DAMPING = 5

    


class ARXInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """


    def __init__(self,
            shm_manager: SharedMemoryManager, 
            robot_ip, 
            robot_port,
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
        self.robot_port = robot_port
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
        self.gripper_scale = 4.0

        # build input queue
        example = {
            'cmd': Command.STOP.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0,
            'data': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )
        
        example = dict()
        example["actual_eef_pose"] = np.zeros(6)
        example["actual_joint_pos"] = np.zeros(7)
        example["actual_joint_vel"] = np.zeros(7)
        example["actual_joint_torque"] = np.zeros(7)
        example["actual_gripper_pos"] = np.zeros(1)
        example["actual_gripper_vel"] = np.zeros(1)
        example["actual_gripper_torque"] = np.zeros(1)
        example["actual_gain"] = np.zeros(1)
        example['robot_receive_timestamp'] = time.time()

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
    
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[ARXPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.RESET_TO_HOME.value,
            'data': None
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
        
    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)
    
    def set_gripper(self, pos):
        message = {
            'cmd': Command.SET_GRIPPER.value,
            'data': pos * self.gripper_scale
        }
        self.input_queue.put(message)

    def reset_to_home(self):
        message = {
            'cmd': Command.RESET_TO_HOME.value,
            'data': None
        }
        self.input_queue.put(message)

    def set_to_damping(self):
        message = {
            'cmd': 'SET_TO_DAMPING',
            'data': None
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start rtde
        robot_ip = self.robot_ip
        robot_port = self.robot_port

        arx_robot = Arx5Client(robot_ip,robot_port)   # Ramdom IP and port

        try:
            if self.verbose:
                print(f"[ARXPositionalController] Connect to robot: {robot_ip}")

            # init pose
            arx_robot.reset_to_home()
            print("[ARXPositionalController] Slave robot is set to initial pose.")

            # main loop
            dt = 1. / self.frequency
            state_data = arx_robot.get_state()
            curr_pose = state_data["ee_pose"]

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )
            gripper_pos = 0.0
            iter_idx = 0
            keep_running = True
            while keep_running:
                # start control iteration
                # t_start = rtde_c.initPeriod()
                t_start = time.perf_counter()

                # send command to robot
                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                pose_command = pose_interp(t_now)
                vel = 0.5
                acc = 0.5
                # assert rtde_c.servoL(pose_command, 
                #     vel, acc, # dummy, not used by ur5
                #     dt, 
                #     self.lookahead_time, 
                #     self.gain)
                # print(pose_command.shape)
                arx_robot.set_ee_pose(pose_command[:6], gripper_pos)
                
                # update robot state
                state = dict()
                # for key in self.receive_keys:
                #     state[key] = np.array(getattr(rtde_r, 'get'+key)())
                state_data = arx_robot.get_state()
                state["actual_eef_pose"] = state_data["ee_pose"]
                state["actual_joint_pos"] = np.hstack((state_data["joint_pos"], np.array(state_data["gripper_pos"])))   
                state["actual_joint_vel"] = np.hstack((state_data["joint_vel"], np.array(state_data["gripper_vel"])))     
                state["actual_joint_torque"] = np.hstack((state_data["joint_torque"], np.array(state_data["gripper_torque"]))) 
                state["actual_gripper_pos"] = state_data["gripper_pos"]
                state["actual_gripper_vel"] = state_data["gripper_vel"]
                state["actual_gripper_torque"] = state_data["gripper_torque"] 
                state['robot_receive_timestamp'] = time.time()
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[RTDEPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.SET_GRIPPER.value:
                        gripper_pos = command['data']
                    elif cmd == Command.RESET_TO_HOME.value:
                        arx_robot.reset_to_home()  
                    elif cmd == Command.SET_TO_DAMPING.value:
                        arx_robot.set_to_damping()
                    else:
                        keep_running = False
                        break
                
                # regulate frequency
                time_remain = 1.0/self.frequency - (time.perf_counter()-t_start)
                if time_remain > 0:
                    time.sleep(time_remain)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[ARXPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            # manditory cleanup
            arx_robot.reset_to_home()
            self.ready_event.set()

            if self.verbose:
                print(f"[ARXPositionalController] Disconnected from robot: {robot_ip}")
