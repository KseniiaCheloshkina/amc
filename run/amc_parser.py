import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat, mat2euler
from mpl_toolkits.mplot3d import Axes3D
import os


##### FUNCTIONS FOR READING DATA

def read_line(stream, idx):
    
    """
    Read line of stream data and split by whitespace. 
    Return: content, index
    """
    if idx >= len(stream):
        line = None
    else:
        line = stream[idx].strip().split()
        idx += 1
    return line, idx


def parse_asf(file_path):
  
    '''
    Read asf parameters and hierarchy and save to dctionary as Joint objects
    '''
    
    with open(file_path) as f:
        content = f.read().splitlines()

    for idx, line in enumerate(content):
    # meta infomation is ignored
        if line == ':bonedata':
            content = content[idx+1:]
            break

    # read joints
    joints = {'root': Joint('root', np.zeros(3), 0, np.zeros(3), [], [])}
    idx = 0
    
    while True:
        
        # check asf format
        # the order of each section is hard-coded
        line, idx = read_line(content, idx)

        if line[0] == ':hierarchy':
            break
        assert line[0] == 'begin'

        line, idx = read_line(content, idx)
        assert line[0] == 'id'

        line, idx = read_line(content, idx)
        assert line[0] == 'name'
        name = line[1]

        line, idx = read_line(content, idx)
        assert line[0] == 'direction'
        direction = np.array([float(axis) for axis in line[1:]])

        # skip length
        line, idx = read_line(content, idx)
        assert line[0] == 'length'
        length = float(line[1])

        line, idx = read_line(content, idx)
        assert line[0] == 'axis'
        assert line[4] == 'XYZ'

        axis = np.array([float(axis) for axis in line[1:-1]])
        
        # read content and save to dictionary
        dof = []
        limits = []

        line, idx = read_line(content, idx)
        if line[0] == 'dof':
            dof = line[1:]
            for i in range(len(dof)):
                line, idx = read_line(content, idx)
                if i == 0:
                    assert line[0] == 'limits'
                    line = line[1:]
                assert len(line) == 2
                mini = float(line[0][1:])
                maxi = float(line[1][:-1])
                limits.append((mini, maxi))

            line, idx = read_line(content, idx)

        assert line[0] == 'end'
        joints[name] = Joint(
          name,
          direction,
          length,
          axis,
          dof,
          limits
        )

    # read hierarchy
    assert line[0] == ':hierarchy'
    line, idx = read_line(content, idx)
    
    assert line[0] == 'begin'
    while True:
        line, idx = read_line(content, idx)

        if line[0] == 'end':
            break
        assert len(line) >= 2
        for joint_name in line[1:]:

            joints[line[0]].children.append(joints[joint_name])
        for nm in line[1:]:
            joints[nm].parent = joints[line[0]]

    return joints


def parse_amc(file_path):
    
    """
    Read amc file and add moton data to list
    """
    
    with open(file_path) as f:
        content = f.read().splitlines()

    for idx, line in enumerate(content):
        if line == ':DEGREES':
            content = content[idx+1:]
            break

    frames = []
    idx = 0
    line, idx = read_line(content, idx)
    assert line[0].isnumeric(), line
    EOF = False
    while not EOF:
        joint_degree = {}
        while True:
            line, idx = read_line(content, idx)
            if line is None:
                EOF = True
                break
            if line[0].isnumeric():
                break
            joint_degree[line[0]] = [float(deg) for deg in line[1:]]
        frames.append(joint_degree)
    
    return frames


###### CLASS FOR EXTRACTING GLOBAL COORDINATES AND ROTATION MATRICES FROM DATA


class Joint:
    
    
    def __init__(self, name, direction, length, axis, dof, limits):
        
        """
        Definition of basic joint. The joint also contains the information of the
        bone between it's parent joint and itself. 
        Save all skeleton info about bone. 

        Parameters
        ---------
        name: Name of the joint defined in the asf file. There should always be one
        root joint. String.

        direction: Default direction of the joint(bone). The motions are all defined
        based on this default pose.

        length: Length of the bone.

        axis: Axis of rotation for the bone.

        dof: Degree of freedom. Specifies the number of motion channels and in what
        order they appear in the AMC file.

        limits: Limits on each of the channels in the dof specification

        """
        
        self.name = name
        self.direction = np.reshape(direction, [3, 1])
        self.length = length
        axis = np.deg2rad(axis)
        self.C = euler2mat(*axis)
        self.Cinv = np.linalg.inv(self.C)
        self.limits = np.zeros([3, 2])
        
        for lm, nm in zip(limits, dof):
            if nm == 'rx':
                self.limits[0] = lm
            elif nm == 'ry':
                self.limits[1] = lm
            else:
                self.limits[2] = lm
                
        self.parent = None
        self.children = []
        self.coordinate = None
        self.norm_coordinate = None
        self.matrix = None
        self.norm_matrix = None

            
    def set_motion(self, motion):
        
        """
        Get parsed amc content and save global rotaion matrices (self.matrix) and global coordinates (self.coordinate) for each bone
        
        Go from root to limbs by hierarchy 
        
        Local rotation L:
            L = CinvMC
        Global rotation matrix T:
            T = L_parent * L   
        self.matrix = T
        
        Global coordinates:
            Coordinate = Coordinate_parent + length * T * direction
             self.coordinate = Coordinate 
        
        """
        
        if self.name == 'root':
            self.coordinate = np.reshape(np.array(motion['root'][:3]), [3, 1])
            rotation = np.deg2rad(motion['root'][3:])
            self.matrix = self.C.dot(euler2mat(*rotation)).dot(self.Cinv)
        else:
            idx = 0
            rotation = np.zeros(3)
            for axis, lm in enumerate(self.limits):
                if not np.array_equal(lm, np.zeros(2)):
                    rotation[axis] = motion[self.name][idx]
                    idx += 1
            rotation = np.deg2rad(rotation)
            self.matrix = self.parent.matrix.dot(self.C).dot(euler2mat(*rotation)).dot(self.Cinv)
            self.coordinate = self.parent.coordinate + self.length * self.matrix.dot(self.direction)
            
        for child in self.children:
            child.set_motion(motion)
            
            
    def normalize(self):
        
        joints = self.to_dict()

        root_coor = joints['root'].coordinate
        root_orient = joints['root'].matrix
        root_orient_inv = np.linalg.inv(root_orient)

        all_bones = list(joints.keys())

        for bone in all_bones:
            ex = joints[bone]
            ex_coordinate = ex.coordinate
            ex_matrix = ex.matrix

            ex_coordinate = ex_coordinate - root_coor
            ex_matrix = np.dot(ex_matrix, root_orient_inv)

            ex.norm_coordinate = ex_coordinate
            ex.norm_matrix = ex_matrix

            
    def draw(self, draw_norm=True):
        
        joints = self.to_dict()
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.set_xlim3d(-50, 10)
        ax.set_ylim3d(-20, 40)
        ax.set_zlim3d(-20, 40)

        xs, ys, zs = [], [], []
        
        for joint in joints.values():
            if draw_norm:
                coord = joint.norm_coordinate
            else:
                coord = joint.coordinate
            xs.append(coord[0, 0])
            ys.append(coord[1, 0])
            zs.append(coord[2, 0])
        plt.plot(zs, xs, ys, 'b.')

        for child in joints.values():
            if child.parent is not None:
                parent = child.parent
                if draw_norm:
                    child_coord = child.norm_coordinate
                    parent_coord = parent.norm_coordinate
                else:
                    child_coord = child.coordinate
                    parent_coord = parent.coordinate               
                xs = [child_coord[0, 0], parent_coord[0, 0]]
                ys = [child_coord[1, 0], parent_coord[1, 0]]
                zs = [child_coord[2, 0], parent_coord[2, 0]]
                plt.plot(zs, xs, ys, 'r')
        plt.show()

            
    def to_dict(self):
        
        ret = {self.name: self}
        for child in self.children:
            ret.update(child.to_dict())
        return ret

    def pretty_print(self):
        
        print('===================================')
        print('joint: %s' % self.name)
        print('direction:')
        print(self.direction)
        print('limits:', self.limits)
        print('parent:', self.parent)
        print('children:', self.children)

        


############ FUNCTION FOR TEST LOAD, TRANSFORM AND DRAWING 


def test_all(data_path='../data'):
    
    """
    Default folder structure:
        data_path/folder_nm/folder_nm.asf
        data_path/folder_nm/folder_nm_{01, 02, ...}.amc
    
    Draw only first amc motion file using asf data.
        
    """
    lv0 = data_path
    lv1s = os.listdir(lv0)
    lv1s.remove('.ipynb_checkpoints')
    for lv1 in lv1s:
        lv2s = os.listdir('/'.join([lv0, lv1]))
        asf_path = '%s/%s/%s.asf' % (lv0, lv1, lv1)
        print('parsing %s' % asf_path)
        joints = parse_asf(asf_path)
        amc_paths = [path for path in lv2s if ('amc' in path) & (path.startswith(lv1))]
        # draw only first
        amc_path = '%s/%s/%s' % (lv0, lv1, amc_paths[0])
        print('parsing %s' % amc_path)
        motions = parse_amc(amc_path)
        joints['root'].set_motion(motions[0])
        joints['root'].normalize()
        joints['root'].draw(draw_norm=True)



if __name__ == '__main__':
    test_all()
