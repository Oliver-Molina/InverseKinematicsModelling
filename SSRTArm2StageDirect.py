from IKEngine import *
import numpy as np
import sympy as sp
import math

def my_DH_trans_matrix(params):
    d, theta, a, alpha = (params[0], params[1], params[2], params[3])

    mat = np.array([[np.cos(theta), -1*np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha),    a*np.cos(theta)],
                    [np.sin(theta), np.cos(theta)*np.cos(alpha),    -1*np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                    [0,             np.sin(alpha),                  np.cos(alpha),                  d],
                    [0,             0,                              0,                              1]])
    
    return mat

def my_joint_transforms(DH_params):
    transforms = []

    transforms.append(np.identity(4))

    for el in DH_params:
        transforms.append(my_DH_trans_matrix(el))
    return np.array(transforms)

def my_trans_EF_eval(DH_params):    
    transforms = my_joint_transforms(DH_params)

    trans_EF = transforms[0]

    for mat in transforms[1:]:

        trans_EF = trans_EF @ mat
    
    trans_EF_cur = trans_EF
                
    return trans_EF_cur

class SSRTArm2StageDirect(Manipulator):
    def __init__(self, links, origin, DH_params, joint_map):
        self.links = links
        self.origin = origin
        self.DH_params = DH_params
        self.joint_map = joint_map

    def getUpdatedJointAngles(self, target=None, orientation=None):
        # If no target return current angles
        if target is None:
            angles = []

            for [i,j] in self.joint_map:
                angles.append(self.DH_params[i][j])
            return np.array(angles)
        
        # Solve for theta1 -> theta3 
        [l1, l2, l3] = self.links
        [x,y,z] = target
        
        theta3 = math.acos((x**2 + y**2 + z**2 - l1**2 - l2**2)/(2*l1*l2))
        theta2 = math.atan2(z, math.sqrt(x**2 + y**2)) - math.atan2(l1 + l2*math.cos(theta3), -l2*math.sin(theta3))
        theta1 = math.atan2(-x, y)

        angles = [theta1, theta2, theta3]

        # Update theta1 -> theta3
        for i in range(len(angles)):
            [j,k] = self.joint_map[i]
            self.DH_params[j,k] = angles[i]

        for i in range(len(angles), len(self.joint_map)):
            [j,k] = self.joint_map[i]
            angles.append(self.DH_params[j,k])


        return angles

    def getUpdatedJointPositions(self, angles=None):
        # Update angles
        if angles is not None:
            for joint_index in range(len(angles)):
                [i, j] = self.joint_map[joint_index]
                self.DH_params[i][j] = angles[joint_index]

        # Compute all positions using forward Kinematics

        # Each link point will be calculated using transform matricies
        # Found with DH_params
        transforms = my_joint_transforms(self.DH_params)

        # Pi = Pi-1 * Pi-2 ... * P0

        P = [np.array([0,0,0])]
        current_transform = np.eye(4)
        for i, trans in enumerate(transforms):
            current_transform = current_transform @ trans
            point = current_transform[:3, 3]
            point = point.reshape(3)
            #point = np.add(point, P[i])
            P.append(point)

        #print(P[-3])
        return P

    def getUpdatedJointOrientations(self, angles=None):
        if angles is not None:
            for joint_index in range(len(angles)):
                [i, j] = self.joint_map[joint_index]
                self.DH_params[i][j] = angles[joint_index]
        
        transforms = my_joint_transforms(self.DH_params)
        trans_EF = np.eye(4)
        orientations = np.array([])

        for trans in transforms:
            trans_EF  = trans_EF @ trans
        
            if trans_EF[0,2] < +1:
                if trans_EF[0,2] > -1:
                    thetaY = math.asin(trans_EF[0,2])
                    thetaX = math.atan2(-trans_EF[1,2], trans_EF[2,2])
                    thetaZ = math.atan2(-trans_EF[0,1], trans_EF[0,0])
                else:  # r02 == -1
                    # Not a unique solution: thetaZ - thetaX = atan2(r10, r11)
                    thetaY = -math.pi / 2
                    thetaX = -math.atan2(trans_EF[1,0], trans_EF[1,1])
                    thetaZ = 0
            else:  # r02 == +1
                # Not a unique solution: thetaZ + thetaX = atan2(r10, r11)
                thetaY = +math.pi / 2
                thetaX = math.atan2(trans_EF[1,0], trans_EF[1,1])
                thetaZ = 0

            orientations = np.append(orientations, [thetaX, thetaY, thetaZ])

        orientations = orientations.reshape(-1,3)

        print(orientations[-1])

        return orientations

    def getManipulatorOrigin(self):
        return self.origin