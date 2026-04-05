import time, sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from simulation.environment import UR5eEnvironment
from simulation.pick_place_sm import PickPlaceStateMachine
from simulation.trajectory_executor import TrajectoryExecutor
from simulation.gripper import VacuumGripper
from simulation.object_detector import ObjectDetector

env = UR5eEnvironment(gui=False)
robot_id = env.get_robot_id()
ee_link = env.get_joint_indices()[-1]
executor = TrajectoryExecutor(env)
gripper = VacuumGripper(robot_id, ee_link)
detector = ObjectDetector(env)

sm = PickPlaceStateMachine(env, executor, gripper, detector)
obj_id = env.get_object_id()

sm.start(obj_id, auto_repeat=False)

print("Running state machine...")
timeout = time.time() + 20.0
while time.time() < timeout:
    status = sm.update()
    if executor.is_running:
        executor.update()
    env.step(1)
    
    if sm.state.name in ['ERROR', 'DONE']:
        print(f"FINISHED IN STATE: {sm.state.name}")
        if status.get('error_msg'):
            print("ERROR:", status['error_msg'])
        break

env.close()
