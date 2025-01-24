from IKEngine import *
import numpy as np
import sympy as sp
import math


# def DH_trans_matrix(params):
    
#     d, theta, a, alpha = (params[0], params[1], params[2], params[3])
    
#     mat = sp.Matrix([[sp.cos(theta), -1*sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha),    a*sp.cos(theta)],
#                     [sp.sin(theta), sp.cos(theta)*sp.cos(alpha),    -1*sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
#                     [0,             sp.sin(alpha),                  sp.cos(alpha),                  d],
#                     [0,             0,                              0,                              1]])
    
#     return mat

# def joint_transforms(DH_params):
#     transforms = []

#     transforms.append(sp.eye(4)) #Assuming the first first joint is at the origin

#     for el in DH_params:

#         transforms.append(DH_trans_matrix(el))

#     return transforms

# def trans_EF_eval(links, joints, DH_params):

#     # Convert to list if it's an ndarray
#     if (isinstance(joints, np.ndarray)):
#         joints = joints.flatten().tolist()
    
#     transforms = joint_transforms(DH_params)

#     trans_EF = transforms[0]

#     for mat in transforms[1:]:

#         trans_EF = trans_EF * mat
    
#     trans_EF_cur = trans_EF
            
#     trans_EF_cur = jacobian_subs(links, joints, trans_EF_cur)
    
#     return trans_EF_cur

# def jacobian_expr(DH_params):

#     transforms = joint_transforms(DH_params)

#     trans_EF = transforms[0]

#     for mat in transforms[1:]:

#         trans_EF = trans_EF * mat

#     pos_EF = trans_EF[0:3,3]

#     J = sp.zeros(6, 6)

#     for joint in range(6):

#         trans_joint = transforms[0]

#         for mat in transforms[1:joint+1]:

#             trans_joint = trans_joint*mat

#         z_axis = trans_joint[0:3,2]

#         pos_joint = trans_joint[0:3,3]

#         Jv = z_axis.cross(pos_EF - pos_joint)

#         Jw = z_axis

#         J[0:3,joint] = Jv
#         J[3:6,joint] = Jw

#     J = sp.simplify(J)
#     return J

# def jacobian_subs(links, joints, jacobian_sym):
    
#     # Convert to list if it's an ndarray
#     if (isinstance(joints, np.ndarray)):
#         joints = joints.flatten().tolist()

#     # Convert to list if it's an ndarray
#     if (isinstance(links, np.ndarray)):
#         links = links.flatten().tolist()
    
#     J_l = jacobian_sym
    
#     J_l = J_l.subs(q1, joints[0])
#     J_l = J_l.subs(q2, joints[1])
#     J_l = J_l.subs(q3, joints[2])
#     J_l = J_l.subs(q4, joints[3])
#     J_l = J_l.subs(q5, joints[4])
#     J_l = J_l.subs(q6, joints[5])

#     J_l = J_l.subs(d1, links[0])
#     J_l = J_l.subs(d2, links[1])
#     J_l = J_l.subs(d3, links[2])
#     J_l = J_l.subs(d4, links[3])
#     J_l = J_l.subs(d5, links[4])
#     J_l = J_l.subs(d6, links[5])
    
#     return J_l

# def joint_limits(joints):
            
#     # Joint 1
#     if (joints[0] < -2*sp.pi/3):
        
#         joints[0] = -2*sp.pi/3
        
#     elif (joints[0] > 2*sp.pi/3):
        
#         joints[0] = 2*sp.pi/3
        
    
#     # Joint 2
#     if (joints[1] < -0.95*sp.pi):
        
#         joints[1] = -0.95*sp.pi
        
#     elif (joints[1] > 0):
        
#         joints[1] = 0
        
#     # Joint 3
#     if (joints[2] < -0.463*sp.pi):
        
#         joints[2] = -0.463*sp.pi
        
#     elif (joints[2] > 0.48*sp.pi):
        
#         joints[2] = 0.48*sp.pi
        
#     # Joint 4
#     if (joints[3] < -0.97*sp.pi):
        
#         joints[3] = -0.97*sp.pi
        
#     elif (joints[3] > 0.97*sp.pi):
        
#         joints[3] = 0.97*sp.pi
            
#     # Joint 5
#     if (joints[4] < -3*sp.pi/2):
        
#         joints[4] = -3*sp.pi/2
        
#     elif (joints[4] > 3*sp.pi/2):
        
#         joints[4] = 3*sp.pi/2
        
#     # Joint 6
#     if (joints[5] < -0.95*sp.pi):
        
#         joints[5] = -0.95*sp.pi
        
#     elif (joints[5] > 0.95*sp.pi):
        
#         joints[5] = 0.95*sp.pi
            
#     return joints

# def i_kine(links_init, joints_init, target, DH_params, error_trace=False, no_rotation=False, joint_lims=True):
    
#     links = links_init
#     joints = joints_init
    
#     xr_desired = target[0:3,0:3]
#     xt_desired = target[0:3,3]

#     print(target)
    
#     x_dot_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
#     e_trace = []
    
#     iters = 0
    
#     print("Finding symbolic jacobian")
    
#     # We only do this once since it's computationally heavy
#     jacobian_symbolic = jacobian_expr(DH_params)
#     print("jacobian_symbolic:", jacobian_symbolic)
    
#     print("Starting IK loop")
    
#     final_xt = 0
#     print("xt_desired: ", xt_desired[0], xt_desired[1], xt_desired[2])

    
#     while(1):
#         # Evaluate current position
#         jac = jacobian_subs(links, joints, jacobian_symbolic)
#         print("jac:", jac)

#         jac = np.array(jac).astype(np.float64)

#         print("jac:", jac)
        
#         trans_EF_cur = trans_EF_eval(links, joints, DH_params)
                
#         trans_EF_cur = np.array(trans_EF_cur).astype(np.float64)
        
        
#         xr_cur = trans_EF_cur[0:3,0:3]
#         xt_cur = trans_EF_cur[0:3,3]
#         print("xt_cur: ", xt_cur[0], xt_cur[1], xt_cur[2])
        
#         final_xt = xt_cur

#         # Evaluate error
                
#         xt_dot = xt_desired - xt_cur
#         print("xt_dot: ", xt_dot[0], xt_dot[1], xt_dot[2])
        
        
#         # Find error rotation matrix
#         R = xr_desired @ xr_cur.T
        
                            
#         # convert to desired angular velocity
#         v = np.arccos((R[0,0] + R[1,1] + R[2,2] - 1)/2)
#         r = (0.5 * np.sin(v)) * np.array([[R[2,1]-R[1,2]],
#                                        [R[0,2]-R[2,0]],
#                                        [R[1,0]-R[0,1]]])
        
        
#         # The large constant just tells us how much to prioritize rotation
#         xr_dot = 200 * r * np.sin(v)
        
#         # use this if you only care about end effector position and not rotation
#         if (no_rotation):
            
#             xr_dot = 0 * r
        
#         xt_dot = xt_dot.reshape((3,1))
                
#         x_dot = np.vstack((xt_dot, xr_dot))
                
#         x_dot_norm = np.linalg.norm(x_dot)
                        
#         if (x_dot_norm > 25):
            
#             x_dot /= (x_dot_norm/25)
            
#         print("x_dot = ", x_dot)

#         x_dot_change = np.linalg.norm(x_dot - x_dot_prev)

#         print("x_dot_change = ", x_dot_change)
                    
#         # This loop now exits if the change in the desired movement stops changing
#         # This is useful for moving close to unreachable points
#         # if (x_dot_change < 0.005):
#         #     print(f"Nearly no change. x_dot_change = {x_dot_change}")
#         #     break
            
#         x_dot_prev = x_dot
            
#         e_trace.append(x_dot_norm)

#         # calculate new position
#         if False:    # Damped
#             Lambda = 0.1 # Should use variable damping coefficient determined by manipulability
#             Alpha = 1
                            
#             joint_change = Alpha * np.linalg.inv(jac.T@jac + Lambda**2*np.eye(DOF)) @ jac.T @ x_dot
#         else:       # Undamped
#             # new = old + inverse_jac * error
#             inverse_jac = np.linalg.pinv(jac)
#             print("inverse_jac: ", inverse_jac)
#             joint_change = inverse_jac @ x_dot

#         print("joint_change: ", joint_change)
        
#         joints += joint_change
        
#         if (joint_lims): joints = joint_limits(joints)

#         if(iters > 1000):
#             print("Maximum iterations.")
#             break
        
#         iters += 1
                
#     print("Done in {} iterations".format(iters))
    
#     print("Final position is:")
#     print(final_xt)
        
#     return (joints, e_trace) if error_trace else joints


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

class SSRTArm(Manipulator):
    def __init__(self, DH_params, joint_map, origin):
        self.DH_params = DH_params
        self.joint_map = joint_map
        self.origin = origin

    def getUpdatedJointAngles(self, target=None, orientation=None):
        # If no target return current angles
        if target is None:
            angles = []

            for [i,j] in self.joint_map:
                angles.append(self.DH_params[i][j])
            return np.array(angles)
        

        # Estimate Jacobian
        DOF = len(self.joint_map)
        current_DH_params = self.DH_params
        curr_angles = self.getUpdatedJointAngles()
        epsilon = 0.1
        jacobian = []

        for joint_index in range(DOF):
            [i,j] = self.joint_map[joint_index]

            prev_angles = curr_angles
            prev_angles[joint_index] -= epsilon
            prev_pos = self.getUpdatedJointPositions(prev_angles)[-1].reshape(3,1)
            prev_ori = self.getUpdatedJointOrientations(prev_angles)[-1].reshape(3,1)

            next_angles = curr_angles
            next_angles[joint_index] += epsilon
            next_pos = self.getUpdatedJointPositions(next_angles)[-1].reshape(3,1)
            next_ori = self.getUpdatedJointOrientations(next_angles)[-1].reshape(3,1)

            dPosdTheta = np.subtract(next_pos, prev_pos) / (2 * epsilon)

            dOridTheta = np.subtract(next_ori, prev_ori) / (2 * epsilon)
            
            dFdTheta = np.append(dPosdTheta, dOridTheta)

            jacobian.append(dFdTheta)

        jacobian = np.array(jacobian).reshape(6,6)

        self.DH_params = current_DH_params

        # Calculate change in angles for current step
        end_effector = my_trans_EF_eval(self.DH_params)
        current_pos = end_effector[:3, 3] # x,y,z component of transform matrix
        pos_error = np.subtract(target, current_pos)
        ori_error = np.array([0,0,0]) if orientation is None else np.subtract(orientation, self.getUpdatedJointOrientations()[-1])

        error = np.append(pos_error, ori_error)
        
        Lambda = 0.01
        inverse_jacobian = np.matmul(np.linalg.inv(np.matmul(jacobian.transpose(), jacobian) + Lambda**2*np.eye(DOF)), jacobian.transpose())

        Alpha_pos = 0.01# * np.array([math.exp(-t/DOF) for t in range(DOF)])
        Alpha_ori = 0.01# * np.array([math.exp((t-DOF)/DOF) for t in range(DOF)])

        pos_change = Alpha_pos * np.matmul(inverse_jacobian, np.append(pos_error, [0,0,0]))
        ori_change = Alpha_ori * np.matmul(inverse_jacobian, np.append([0,0,0], ori_error))
        #print(math.sqrt(sum(i**2 for i in pos_error)), math.sqrt(sum(i**2 for i in ori_error)))

        delta_angles = np.add(pos_change, ori_change)

        new_angles = np.add(curr_angles, delta_angles)

        #if over limit restrict
        return new_angles

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
            point = np.array(current_transform[:3, 3])
            point = point.reshape(3)
            #point = np.add(point, P[i])
            P.append(point)

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

        return orientations.reshape(-1,3)

    def getManipulatorOrigin(self):
        return self.origin