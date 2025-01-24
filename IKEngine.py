import numpy as np
import math
from abc import ABC, abstractmethod

class Manipulator:
    REQUIRED_METHODS = ['getUpdatedJointAngles', 'getUpdatedJointPositions', 'getUpdatedJointOrientations', 'getManipulatorOrigin']
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for method in cls.REQUIRED_METHODS:
            if not hasattr(cls, method):
                raise TypeError(f"{cls.__name__} must implement '{method}'.")


class IKEngine:
    def __init__(self, manipulators:list[Manipulator]):
        self.manipulators = manipulators

    def getSkeleton(self, targets:list, orientations:list):
        skeleton = np.array([[]])   # skeleton is an adjacency list of connection points used to construct a visual

        if len(targets) > len(self.manipulators):
            
            print("length of targets is ", len(targets))
            print("length of manipulators is ", len(self.manipulators))
            print("getSKeleton: More targets than manipulators, ignoring excess arguments.")

        for i, manipulator in enumerate(self.manipulators):
            # add each link to skeleton
            target = targets[i] if i < len(targets) else None
            orientation = orientations[i] if i < len(orientations) else None
            manipulator_angles = manipulator.getUpdatedJointAngles(target, orientation) # Use current angles if no target supplied

            manipulator_points = manipulator.getUpdatedJointPositions(manipulator_angles)

            manipulator_orientations = manipulator.getUpdatedJointOrientations(None)

            #print(manipulator_orientations * 180/math.pi)
            
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

    