import numpy as np
import pybullet as p
import pybullet_data


class KukaEnv:
    '''
    Interface class for kuka environment
    '''

    EPS = 0.1

    def __init__(self, GUI=False):
        '''
        :param GUI: True if user wants to show the GUI
        '''
        self.kuka_file = "kuka_iiwa/model.urdf"

        if GUI:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        
        target = p.getDebugVisualizerCamera()[11]
        p.resetDebugVisualizerCamera(
            cameraDistance=1.1,
            cameraYaw=90,
            cameraPitch=-25,
            cameraTargetPosition=[target[0], target[1], 0.7])

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.kukaId = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

        self.config_dim = p.getNumJoints(self.kukaId)
        self.pose_range = [(p.getJointInfo(self.kukaId, jointId)[8], p.getJointInfo(self.kukaId, jointId)[9]) for
                           jointId in
                           range(p.getNumJoints(self.kukaId))]
        self.bound = np.array(self.pose_range).T.reshape(-1)

        self.kukaEndEffectorIndex = self.config_dim-1

        p.setGravity(0, 0, -10)

        p.stepSimulation()

    def set_config(self, c, kukaId=None):
        if kukaId is None:
            kukaId = self.kukaId
        for i in range(p.getNumJoints(kukaId)):
            p.resetJointState(kukaId, i, c[i])
        p.performCollisionDetection()

    def plot(self, path, make_gif=False):
        path = np.array(path)
        self.set_config(path[0])

        goal_kuka = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
                              flags=p.URDF_IGNORE_COLLISION_SHAPES)
        self.set_config(path[-1], goal_kuka)

        gifs = []
        current_state_idx = 0

        while True:
            disp = path[current_state_idx + 1] - path[current_state_idx]

            d = self.distance(path[current_state_idx], path[current_state_idx + 1])
            K = int(d / self.EPS)
            new_kuka = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True,
                                  flags=p.URDF_IGNORE_COLLISION_SHAPES)
            for data in p.getVisualShapeData(new_kuka):
                color = list(data[-1])
                color[-1] = 0.5
                p.changeVisualShape(new_kuka, data[1], rgbaColor=color)
            for k in range(0, K):
                c = path[current_state_idx] + k * 1. / K * disp
                self.set_config(c, new_kuka)
                p.performCollisionDetection()

                image = p.getCameraImage(width=1080, height=900, lightDirection=[1, 1, 1], shadow=1,
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
                if make_gif:
                    gifs.append(image)

            current_state_idx += 1

            if current_state_idx == len(path) - 1:
                break

        return gifs

    # ===========================Collision Checking=================================

    def valid_state(self, state):
        return (state >= np.array(self.pose_range)[:, 0]).all() and \
               (state <= np.array(self.pose_range)[:, 1]).all()

    def is_state_free(self, state):
        if not self.valid_state(state):
            return False

        self.set_config(state)
        if len(p.getContactPoints(self.kukaId)) == 0:
            return True
        else:
            return False

    def is_edge_free(self, state, new_state):
        self.k = 0
        assert state.size == new_state.size

        if not self.valid_state(state) or not self.valid_state(new_state):
            return False
        if not self.is_state_free(state) or not self.is_state_free(new_state):
            return False

        disp = new_state - state

        d = self.distance(state, new_state)
        K = int(d / self.EPS)
        for k in range(0, K):
            c = state + k * 1. / K * disp
            if not self.is_state_free(c):
                return False
        return True

    # =============================Sampling==========================================

    def uniform_sample(self, n=1):
        '''
        Uniformlly sample in the configuration space
        '''
        sample = np.random.uniform(np.array(self.pose_range)[:, 0], np.array(self.pose_range)[:, 1], size=(n, self.config_dim))
        if n==1:
            return sample.reshape(-1)
        else:
            return sample

    def distance(self, from_state, to_state):
        '''
        Distance metric
        '''

        to_state = np.maximum(to_state, np.array(self.pose_range)[:, 0])
        to_state = np.minimum(to_state, np.array(self.pose_range)[:, 1])
        diff = np.abs(to_state - from_state)

        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def interpolate(self, from_state, to_state, ratio):
        diff = to_state - from_state

        new_state = from_state + diff * ratio
        new_state = np.maximum(new_state, np.array(self.pose_range)[:, 0])
        new_state = np.minimum(new_state, np.array(self.pose_range)[:, 1])

        return new_state

    def in_goal_region(self, state):
        '''
        Return whether a state(configuration) is in the goal region
        '''
        return self.distance(state, self.goal_state) < self.EPS and \
               self._is_state_free(state)
