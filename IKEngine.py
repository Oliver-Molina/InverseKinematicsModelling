import numpy as np
import math
from abc import ABC, abstractmethod


class IKEngine:
    REQUIRED_METHODS = ['getSkeleton']
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for method in cls.REQUIRED_METHODS:
            if not hasattr(cls, method):
                raise TypeError(f"{cls.__name__} must implement '{method}'.")

class AnalyticalManipulator:
    REQUIRED_METHODS = ['getUpdatedJointAngles', 'getUpdatedJointPositions', 'getManipulatorOrigin']
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for method in cls.REQUIRED_METHODS:
            if not hasattr(cls, method):
                raise TypeError(f"{cls.__name__} must implement '{method}'.")



class AnalyticalIKEngine(IKEngine):
    manipulators = []
    def __init__(self, manipulators:list):
        self.manipulators = manipulators

    def getSkeleton(self, targets:list):
        skeleton = np.array([[]])   # skeleton is an adjacency list of connection points used to construct a visual

        if len(targets) > len(self.manipulators):
            
            print("length of targets is ", len(targets))
            print("length of manipulators is ", len(self.manipulators))
            print("getSKeleton: More targets than manipulators, ignoring excess arguments.")

        for i, manipulator in enumerate(self.manipulators):
            # add each link to skeleton
            manipulator_angles = manipulator.getUpdatedJointAngles(targets[i] if i < len(targets) else None) # Use current angles if no target supplied
            #manipulator_angles = [0, math.pi/4, math.pi/4]

            manipulator_points = manipulator.getUpdatedJointPositions(manipulator_angles)
            
            manipulator_skeleton = []
            for i in range(1, len(manipulator_points)):
                manipulator_skeleton.append([manipulator_points[i - 1], manipulator_points[i]])

            
            for j in range(len(manipulator_skeleton)):
                for k in range(len(manipulator_skeleton[j])):
                    manipulator_skeleton[j][k] = np.add(manipulator_skeleton[j][k], manipulator.getManipulatorOrigin())

            # add legs to skeleton
            skeleton = np.append(skeleton, manipulator_skeleton)
        
        # Enforce correct dimensions
        skeleton = skeleton.reshape(-1, 2, 3)
        return skeleton

class KinematicsChain:
    def __init__(self, dh_params):
        self.dh_params = dh_params


class NumericalIKEngine(IKEngine):
    def __init__(self, manipulators:list[KinematicsChain]):
        self.manipulators = manipulators

    def getSkeleton():
        pass

    