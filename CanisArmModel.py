from IKEngine import *
import math

# Restricts input to within +-1 i.e domain of asin and acos
def restrictInverseSinDomain(input):
    return math.copysign(min(1, abs(input)), input)



def generate2JointSinglePlainArm(x, y, l1, l2):
    ratio = restrictInverseSinDomain((x**2 + y**2 - l2**2 - l1**2)/(2*l2*l1))
    theta2 = math.acos(ratio)
    theta1 = math.atan2(y, x) - math.atan2((l2*math.sin(theta2)), (l2*math.cos(theta2) + l1))
    return [theta1, theta2]

def update2JointSinglePlainArm(desiredx, desiredy, l1 , l2):
    theta1, theta2 = generate2JointSinglePlainArm(desiredx, desiredy, l1 , l2)

    x1 = l1*math.cos(theta1)
    y1 = l1*math.sin(theta1)

    x2 = x1 + l2*math.cos(theta2 + theta1)
    y2 = y1 + l2*math.sin(theta2 + theta1)

    return [[x1,y1], [x2,y2]]

def generate3Joint3DArmAnglesZ(x, y, z, l1, l2):
    theta3= math.acos((x**2 + y**2 + z**2 - l2**2 - l1**2)/(2*l2*l1))
    theta2 = math.atan2(z, math.sqrt(x**2 + y**2)) - math.atan2((l2*math.sin(theta3)), (l2*math.cos(theta3) + l1))
    theta1 = math.atan2(y, x)
    return [theta1, theta2, theta3]

def generate3Arm3DAnglesX(x, y, z, l1, l2, l3):
    # Given a 3 link 3 DOF arm with shoulder l1 oriented about the X axis and arm segments l2 and l3 both normal to l1
    # theta1 can be solved for using a triangle with lengths l1, h = magnitude(y,z), 
    # and unknowns theta_r = ?, and r = sqrt(h^2 - l1^2)
    # After setting up the triangle

    # The copy sign trash is to fix output of acos since it doesn't account for quadrant as does atan2, TODO: find a better solution
    correctsign = -math.copysign(1, y)*math.copysign(1, z) 
    shoulderToArmRatio = restrictInverseSinDomain(l1/math.sqrt(y**2 + z**2))
    theta1 = correctsign*math.acos(shoulderToArmRatio) + math.atan2(z,y)   

    print(math.acos(shoulderToArmRatio) * 180/math.pi, math.atan2(z,y) * 180/math.pi)

    # Considering a cylindrical coordinates reference frame about the x axis with 
    # origin centered on the end of l1 it is much easier to solve theta2 and theta3
    yadj = y - l1*math.cos(theta1)
    zadj = z - l1*math.sin(theta1)

    # In the x,r plane formed by links l2 and l3 we can now solve for our angles as if 
    # it were merely a planar 2 joint arm 
    r = math.sqrt(yadj**2 + zadj**2)

    theta2, theta3 = -correctsign * np.array(generate2JointSinglePlainArm(x, r, l2, l3))

    return [theta1, theta2, theta3]
    
class XOriented3DOF3LinkArm(Manipulator):
    def __init__(self, links, joints, origin):
        self.links = links
        self.joints = joints
        self.origin = origin

    def getUpdatedJointAngles(self, target=None, orientation=None):
        if target is not None:
            [x,y,z] = target
            [l1,l2,l3] = self.links
            self.joints = generate3Arm3DAnglesX(x, y, z, l1, l2, l3)

        return self.joints

    def getUpdatedJointPositions(self, angles=None):
        # Compute all positions using forward Kinematics
        [l0, l1, l2] = self.links
        [theta1, theta2, theta3] = self.angles if angles is None else angles

        # Simplified forwards kinematics equations for each link
        P0 = np.array(self.origin) 

        P1 = np.add(P0, [0, math.cos(theta1)*l0, math.sin(theta1)*l0])

        P2 = np.add(P1, [math.cos(theta2)*l1, -math.sin(theta1)*math.sin(theta2)*l1, math.cos(theta1)*math.sin(theta2)*l1])

        P3 = np.add(P2, [math.cos(theta3 + theta2)*l2, -math.sin(theta1)*math.sin(theta3 + theta2)*l2, math.cos(theta1)*math.sin(theta3 + theta2)*l2])

        P = [P0, P1, P2, P3]

        print(P0, P1, P2, P3)

        return P
    
    def getUpdatedJointOrientations(self, angles=None):
        return np.array([0,0,0])

    def getManipulatorOrigin(self):
        return self.origin
    
    def getTargetPosition(self):
        return self.getUpdatedJointPositions()[-1]
    
    def getTargetOrientation(self):
        return self.getUpdatedJointOrientations()[-1]