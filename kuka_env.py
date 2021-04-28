import numpy as np
import pybullet as p
import pybullet_data


class KukaEnv:
    '''
    Interface class for maze environment
    '''

    EPS = 0.1

    def __init__(self, GUI=False):
        '''
        :param GUI:
        :param kuka_file:
        '''

        self.dim = 3
        self.kuka_file = "kuka_iiwa/model.urdf"

        self.collision_check_count = 0

        self.maps = {}
        self.episode_i = 0
        self.collision_point = None

        if GUI:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.kukaId = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        p.stepSimulation()

        target = p.getDebugVisualizerCamera()[11]
        p.resetDebugVisualizerCamera(
            cameraDistance=1.1,
            cameraYaw=90,
            cameraPitch=-25,
            cameraTargetPosition=[target[0], target[1], 0.7])

        self.config_dim = p.getNumJoints(self.kukaId)
        self.pose_range = [(p.getJointInfo(self.kukaId, jointId)[8], p.getJointInfo(self.kukaId, jointId)[9]) for
                           jointId in
                           range(p.getNumJoints(self.kukaId))]
        self.bound = np.array(self.pose_range).T.reshape(-1)

        self.kukaEndEffectorIndex = self.config_dim-1

        p.setGravity(0, 0, -10)

    def set_random_init_goal(self):
        while True:
            points = self.sample_n_free_points(n=2)
            init, goal = points[0], points[1]
            if np.sum(np.abs(init - goal)) != 0:
                break
        self.init_state, self.goal_state = init, goal

    def sample_n_free_points(self, n):
        samples = []
        for i in range(n):
            while True:
                sample = self.uniform_sample()
                if self._point_in_free_space(sample):
                    samples.append(sample)
                    break
        return samples

    def set_config(self, c, kukaId=None):
        if kukaId is None:
            kukaId = self.kukaId
        for i in range(p.getNumJoints(kukaId)):
            p.resetJointState(kukaId, i, c[i])
        p.performCollisionDetection()

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
               self._state_fp(state)

    def step(self, state, action=None, new_state=None, check_collision=True):
        '''
        Collision detection module
        '''
        # must specify either action or new_state
        if action is not None:
            new_state = state + action

        new_state = np.maximum(new_state, np.array(self.pose_range)[:, 0])
        new_state = np.minimum(new_state, np.array(self.pose_range)[:, 1])

        action = new_state - state

        if not check_collision:
            return new_state, action

        done = False
        no_collision = self._edge_fp(state, new_state)
        if no_collision and self.in_goal_region(new_state):
            done = True

        return new_state, action, no_collision, done

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

    # =====================internal collision check module=======================

    def _valid_state(self, state):
        return (state >= np.array(self.pose_range)[:, 0]).all() and \
               (state <= np.array(self.pose_range)[:, 1]).all()

    def _point_in_free_space(self, state):
        if not self._valid_state(state):
            return False

        self.set_config(state)
        if len(p.getContactPoints(self.kukaId)) == 0:
            return True
        else:
            self.collision_point = state
            return False

    def _state_fp(self, state):
        self.collision_check_count += 1
        return self._point_in_free_space(state)

    def _iterative_check_segment(self, left, right):
        if np.sum(np.abs(left - left)) > 0.1:
            mid = (left + right) / 2.0
            self.k += 1
            if not self._state_fp(mid):
                self.collision_point = mid
                return False
            return self._iterative_check_segment(left, mid) and self._iterative_check_segment(mid, right)

        return True

    def _edge_fp(self, state, new_state):
        self.k = 0
        assert state.size == new_state.size

        if not self._valid_state(state) or not self._valid_state(new_state):
            return False
        if not self._state_fp(state) or not self._state_fp(new_state):
            return False

        disp = new_state - state

        d = self.distance(state, new_state)
        K = int(d / self.EPS)
        for k in range(0, K):
            c = state + k * 1. / K * disp
            if not self._state_fp(c):
                return False
        return True
