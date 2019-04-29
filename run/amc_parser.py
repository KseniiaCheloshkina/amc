import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat, mat2euler
from mpl_toolkits.mplot3d import Axes3D
import os
import copy
import pandas as pd


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
    if '.ipynb_checkpoints' in lv1s:
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
    
    return joints, motions



############ FUNCTION FOR COLLECTION OF JOINTS AND MOTIONS DATA OF ONE AMC FRAME INTO DATAFRAME 

def get_one_frame_data(joints, motion):
    
    """
    Parse one frame data to dataframe
    
    """
    
    joints_nm = list(joints.keys())
    
    all_joints = []
    all_joints_nm = []
    
    for joint_nm in joints_nm:
        
        joint = joints[joint_nm]
        
        # parameters from self
        coord = joint.coordinate.ravel().tolist()
        coord_nm = [joint.name + '_coord_' + str(i) for i in range(3)]

        norm_coord = joint.norm_coordinate.ravel().tolist()
        norm_coord_nm = [joint.name + '_norm_coord_' + str(i) for i in range(3)]

        ax, ay, az = mat2euler(joint.matrix)
        norm_ax, norm_ay, norm_az = mat2euler(joint.norm_matrix)

        l = joint.length
        
        # add to storage
        all_joints.extend(coord)
        all_joints_nm.extend(coord_nm)
        
        all_joints.extend(norm_coord)
        all_joints_nm.extend(norm_coord_nm)
        
        all_joints.extend([ax, ay, az, norm_ax, norm_ay, norm_az, l])
        nm_rest = ['angle_0', 'angle_1', 'angle_2', 'norm_angle_0', 'norm_angle_1', 'norm_angle_2', 'length']
        all_joints_nm.extend([str(joint.name) + "_"+ i for i in nm_rest])
        
        
    # parameters from motion (local)
    all_pars = []
    all_nms = []

    for bone, pars in motion.items():
        all_pars.extend(pars)
        all_nms.extend([bone + "_motion_" + str(i) for i in range(len(pars))])

    all_joints.extend(all_pars)
    all_joints_nm.extend(all_nms)

    df = pd.DataFrame(all_joints).T
    df.columns = all_joints_nm
        
    return df


############ FUNCTION FOR DATAFRAME CREATION FROM ALL AMC FILES 

def create_dataframe_from_data(all_amc_joints, all_motions):
    
    """
    Get full df of all amc frames
    """
    
    df_all = pd.DataFrame()

    amc_paths = list(all_amc_joints.keys())

    for amc_path in amc_paths:
        joints = all_amc_joints[amc_path]
        motions = all_motions[amc_path]

        n_frames = len(joints)

        for i in range(n_frames):

            j_frame = joints[i]
            motion = motions[i]

            df = get_one_frame_data(j_frame, motion)
            df['frame'] = i
            df['amc_path'] = amc_path

            df_all = pd.concat([df_all, df], axis=0)
            
    return df_all
    

############ FUNCTION TO GET ALL DATA FOR A PERSON  

def get_all_data(person_folder):
    
    """
    Default folder structure:
        person_folder/person_folder.asf
        person_folder/person_folder_{01, 02, ...}.amc
        
    Input:
    - person_folder: folder of one person data. Must contain one .asf file and .amc file/files
    
    Output:
    - df: dataframe with one row per frame. Each row contains local motion data, global coordinates and angles, normalized global coordinates and angles, length of bones
        
    """
    
    # find files
    person_files = os.listdir(person_folder)
    amc_paths = []
    asf_paths = []

    for fl in person_files:
        if "asf" in fl:
            asf_paths.append(fl)
        elif 'amc' in fl:
            amc_paths.append(fl)

    if len(asf_paths) > 1:
        raise Exception("More than one asf file found")
    else:
        asf_path = person_folder + asf_paths[0]
    
    # initialize storage
    all_amc_joints = {}
    all_motions = {}
    
    # parse asf
    joints = parse_asf(asf_path)
    
    # parse amc
    for amc_path in amc_paths:
        
        amc_joints = []
        # list of dictionaries
        motions = parse_amc(person_folder + amc_path)
        
        for motion in motions:
            
            joints_new = copy.deepcopy(joints)
            
            # dictionary
            joints_new['root'].set_motion(motion)
            joints_new['root'].normalize()
            
            amc_joints.append(joints_new)
        
        all_amc_joints[amc_path] = amc_joints
        all_motions[amc_path] = motions
    
    
    df = create_dataframe_from_data(all_amc_joints, all_motions)
    
    df['person'] = asf_path.replace(person_folder, "").replace(".asf", "")

    return df


if __name__ == '__main__':
    test_all()
