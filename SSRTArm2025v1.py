from IKEngine import *
import numpy as np
import math

def create_DH_trans_matrix(params):
    theta, alpha, a, d = (params[0], params[1], params[2], params[3])

    mat = np.array([[np.cos(theta), -1*np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha),    a*np.cos(theta)],
                    [np.sin(theta), np.cos(theta)*np.cos(alpha),    -1*np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                    [0,             np.sin(alpha),                  np.cos(alpha),                  d],
                    [0,             0,                              0,                              1]])
    
    return mat

def compute_transforms(DH_params):
    transforms = []

    transforms.append(np.identity(4))

    for el in DH_params:
        transforms.append(create_DH_trans_matrix(el))
    return np.array(transforms)

def evaulate_end_effector(DH_params):    
    transforms = compute_transforms(DH_params)

    trans_EF = transforms[0]

    for mat in transforms[1:]:

        trans_EF = trans_EF @ mat
    
    trans_EF_cur = trans_EF
                
    return trans_EF_cur

def rotation_matrix_zyx(orientation:list):
    [thetax, thetay, thetaz] = orientation
    
    # Rotation matrix about the z-axis
    Rz = np.array([
        [math.cos(thetaz), -math.sin(thetaz), 0],
        [math.sin(thetaz),  math.cos(thetaz), 0],
        [0,                0,                1]
    ])
    
    # Rotation matrix about the y-axis
    Ry = np.array([
        [math.cos(thetay),  0, math.sin(thetay)],
        [0,                1, 0],
        [-math.sin(thetay), 0, math.cos(thetay)]
    ])
    
    # Rotation matrix about the x-axis
    Rx = np.array([
        [1, 0,                0],
        [0, math.cos(thetax), -math.sin(thetax)],
        [0, math.sin(thetax),  math.cos(thetax)]
    ])
    
    # Combined rotation matrix: ZYX order
    R = Rx @ Ry @ Rz  # Matrix multiplication in ZYX order

    return R

class SSRTArm2025v1():
    NUM_OF_JOINTS = 6

    def __validate_list(self, list_arg):
        if isinstance(list_arg, list):
            return list_arg
        if isinstance(list_arg, np.ndarray):
            return list_arg.flatten().tolist()
        raise ValueError(f"list_arg type is not list type")
        
    def __init__(self, link_lengths, joint_defaults, joint_limits, position_boundaries):
        self.__validate_list(link_lengths)
        self.__validate_list(joint_defaults)
        self.__validate_list(joint_limits)
        self.__validate_list(position_boundaries)

        self.links = link_lengths
        self.joints = joint_defaults
        self.joint_limits = joint_limits
        self.position_boundaries = position_boundaries

        # Define DH Parameter Table
        # d, theta, a, alpha
        self.DH_params = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [link_lengths[0], 0, 0, 0],
                                [link_lengths[1], 0, 0, 0],
                                [0, 0, 0, 0],
                                [link_lengths[2], 0, 0, 0]])
        # theta, alpha, a, d
        self.DH_params = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, link_lengths[0]],
                                [0, 0, 0, link_lengths[1]],
                                [0, 0, 0, 0],
                                [0, 0, 0, link_lengths[2]]])
        # Define Joint Mapping
        self.joint_map = np.array([[0, 0], [1, 1], [2, 1], [3, 0], [4, 1], [5, 0]])

    # Returns a list of positions for each joint and the end effector
    def GetPositions(self, current_angles=None):
        # Compute all positions using forward Kinematics

        # Each link point will be calculated using transform matricies
        # Found with DH_params
        transforms = compute_transforms(self.GenerateDHParams(current_angles))
        # Pi = Pi-1 * Pi-2 ... * P0

        P = [np.array([0,0,0])]
        current_transform = np.eye(4)
        for i, trans in enumerate(transforms):
            current_transform = current_transform @ trans
            point = current_transform[:3, 3]
            point = point.reshape(3)
            P.append(point)

        return P

    # Returns a list of orientations for each joint and the end effector
    def GetOrientations(self, current_angles=None):
        transforms = compute_transforms(self.GenerateDHParams(current_angles))
        trans_EF = np.eye(4)
        orientations = np.array([])

        for trans in transforms:
            trans_EF  = trans_EF @ trans
        
            if abs(trans_EF[2, 0]) < 1:  # Standard case, no gimbal lock
                thetaZ = math.atan2(trans_EF[1, 0], trans_EF[0, 0])  # θz (yaw)
                thetaY = math.asin(-trans_EF[2, 0])         # θy (pitch)
                thetaX = math.atan2(trans_EF[2, 1], trans_EF[2, 2]) # θx (roll)
            else:  # Gimbal lock case
                thetaZ = math.atan2(-trans_EF[0, 1], trans_EF[1, 1])
                thetaY = math.pi / 2 if trans_EF[2, 0] < 0 else -math.pi / 2
                thetaX = 0  # Roll is indeterminate in gimbal lock

            orientations = np.append(orientations, [thetaX, thetaY, thetaZ])

        orientations = orientations.reshape(-1,3)

        return orientations

    # Returns a list of angles for the joints
    def GetAngles(self):
        angles = []
        for [i,j] in self.joint_map:
                angles.append(self.DH_params[i][j])
        
        return np.array(angles)

    # Method for validating the joint angles against saved limitations
    def ValidateJointAngles(self, angles):
        # Joint limits will be added at a later date
        if angles is None:
            return False
        
        return True
        
    # Method for Updating joint angles
    def UpdateJointAngles(self, new_angles=None):
        if not self.ValidateJointAngles(new_angles):
            return False
        
        for joint_index in range(len(new_angles)):
            [i, j] = self.joint_map[joint_index]
            self.DH_params[i][j] = new_angles[joint_index]

        return True
    
    def GenerateDHParams(self, current_angles=None):
        if current_angles is None:
            current_angles = self.GetAngles()

        end_effector_index = len(current_angles)
        DH_params = self.DH_params[:end_effector_index]
        if current_angles is not None:
            for joint_index in range(end_effector_index):
                [i, j] = self.joint_map[joint_index]
                DH_params[i][j] = current_angles[joint_index]

        return DH_params
        

    # Method for evaluating arm transform up to the joint_n where n is the number of angles supplied
    def EvaluateTransforms(self, current_angles=None):
        DH_params = self.GenerateDHParams(current_angles)
        
        return evaulate_end_effector(DH_params)

    # Method for updating the positon of the arm while maintaining the current orientation of the wrist
    def GeneratePosition(self, new_position, current_angles=None):
        # Use stored angles if no input specified
        if current_angles is None:
            current_angles = self.GetAngles()
        
        # Step 1: Calculate wrist position (theta1 -> theta3) by subtracting the claw vector from the target position
        orientation = self.GetOrientations(current_angles)[-1]
        
        [l1, l2, l3] = self.links
        
        claw_default = np.atleast_2d([0, 0, l3]).T

        Rot_desired = rotation_matrix_zyx(orientation)

        claw_desired = Rot_desired @ claw_default

        wrist_position = new_position - claw_desired.T.flatten()

        [x,y,z] = wrist_position
        
        # Step 2: Calculate wrist position using 3DOF Robot Arm Equations
        theta3 = math.acos((x**2 + y**2 + z**2 - l1**2 - l2**2)/(2*l1*l2))
        theta2 = math.atan2(z, math.sqrt(x**2 + y**2)) - math.atan2(l1 + l2*math.cos(theta3), -l2*math.sin(theta3))
        theta1 = math.atan2(-x, y)

        current_angles[0] = theta1
        current_angles[1] = theta2
        current_angles[2] = theta3

        # Step 3: Fix orientation by correcting claw angles
        return self.GenerateOrientation(orientation, current_angles)

    # Method for updating the orientation of the arm while keeping the current position of the wrist
    def GenerateOrientation(self, new_orientation, current_angles=None):
        if current_angles is None:
            current_angles = self.GetAngles()
        
        # Compute theta4 -> theta6 to match desired orientation
        # Equate current claw vector based on theta1->6 (with 1->3 known) with desired claw vector
        # Rot(1 -> 3) * claw_required = Rot_desired * claw_default
        # Solve for claw_required
        # claw_required = Rot(1 ->3)^-1 * Rot_desired * claw_default
        # Inverse of rotational matrix is simply its transpose
        # claw_required = Rot(1 ->3)^T * Rot_desired * claw_default
        # Divide by l3 to normalize and solve for angles

        # Step 1: Solve for required claw rotational matrix
        [l1, l2, l3] = self.links

        claw_default = np.atleast_2d([0, 0, l3]).T

        Rot_desired = rotation_matrix_zyx(new_orientation)

        claw_desired = Rot_desired @ claw_default

        Rot_1_3 = self.EvaluateTransforms(current_angles[:3])[:3, :3]
        
        claw_required = np.transpose(Rot_1_3) @ claw_desired

        # Step 2: Calculate theta4 & theta5 using the rotational matrix pitch and yaw
        [Vx, Vy, Vz] = claw_required / l3

        cosTheta5 = Vz
        sinTheta5 = math.sqrt(1 - (Vz)**2)

        if abs(sinTheta5) == 0:
            theta4 = 0
            theta5 = 0
        else:
            cosTheta4 = -Vy/sinTheta5
            sinTheta4 = Vx/sinTheta5

            theta4 = math.atan2(sinTheta4, cosTheta4)
            theta5 = math.atan2(sinTheta5, cosTheta5)

        current_angles[3] = theta4
        current_angles[4] = theta5

        # Step 3: Calculate the theta6 to correct claw roll
        Rot_required = np.transpose(Rot_1_3) @ Rot_desired

        theta6 = math.acos(Rot_required[0,0])

        current_angles[5] = theta6

        return current_angles

    # Method for updating the position and orientation of the rover arm
    # if new_position is None the wrist position will remain fixed and only the claw will move
    # if new_orientation is None the wrist will maintain its current orientation
    # if both are None the the current angles will be returned
    def GenerateJointAngles(self, new_position=None, new_orientation=None):
        if new_position is None and new_orientation is None:
            return self.GetAngles()
        elif new_position is None:
            return self.GenerateOrientation(new_orientation)
        elif new_orientation is None:
            return self.GeneratePosition(new_position)
        else:
            angles = self.GenerateOrientation(new_orientation)
            angles = self.GeneratePosition(new_position, angles)
            return angles

class SSRTArmWrapper(SSRTArm2025v1, Manipulator):

    def getUpdatedJointAngles(self, target=None, orientation=None):
        try:
            angles = self.GenerateJointAngles(target, orientation)
            self.UpdateJointAngles(angles)
        except Exception as e:
            print(e)
            pass

        return self.GetAngles()
    
    def getUpdatedJointPositions(self, angles=None):
        self.UpdateJointAngles(angles)
        positions = self.GetPositions()
        return positions

    def getUpdatedJointOrientations(self, angles=None):
        self.UpdateJointAngles(angles)
        return self.GetOrientations()

    def getManipulatorOrigin(self):
        return np.array([0,0,0])
    

# With orientation disabled the engine jitters

# Any time we calculate ourself based off ourself we open the 
# door to jitter. This is a major problem
# I'm not sure what to do about this just yet
# If there is more than one solution causing jitter, the "simple"
# fix should be to always only take the solution with the minimal change