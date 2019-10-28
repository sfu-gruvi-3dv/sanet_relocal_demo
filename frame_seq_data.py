import functools

import numpy as np
import copy
import json
import core_3dv.camera_operator as cam_opt

def K_from_frame(frame):
    intrisic = frame['camera_intrinsic']
    return np.asarray([[intrisic[0], 0, intrisic[2]], [0, intrisic[1], intrisic[3]], [0, 0, 1]], dtype=np.float32)


class FrameSeqData:
    """
    Frame Sequences collection

    dict:
        'file_name': image file path
        'id': frame_idx
        'extrinsic_Tcw': transform matrix from world to camera
        'camera_intrinsic': camera intrinsic for the frame
        'timestamp': frame time_stamp
        'frame_dim': image dimension (h, w)
        'depth_file_name': optional, depth file path
        'tag': optional

    example:
    # >>> frames = FrameSeqData()
    # >>> frames.load_json('keyframe.json')       # Load from JSON
    # >>> frame.dump_to_json('keyframe.json')     # Save to json
    """

    def __init__(self, json_file_path=None, tag=None):
        self.frames = []
        self.timestamp_cache = None
        if json_file_path != None:
            self.load_json(json_file_path)
            self.__build_timestamp_cache__()
        self.tag = tag

    def __len__(self):
        return len(self.frames)

    def __build_timestamp_cache__(self):
        ts_list = []
        for frame in self.frames:
            _, timestamp = self.get_idx_n_timestamp(frame)
            ts_list.append(timestamp)
        self.timestamp_cache = np.asarray(ts_list)

    def frame_height(self):
        return self.frames[0]['frame_dim'][0] if len(self.frames) > 0 else -1

    def frame_width(self):
        return self.frames[0]['frame_dim'][1] if len(self.frames) > 0 else -1

    def sort_by_frame_idx(self):
        def idx_comparator(a, b):
            if a['id'] < b['id']:
                return -1
            elif a['id'] > b['id']:
                return 1
            else:
                return 0
        self.frames = sorted(self.frames, key=functools.cmp_to_key(idx_comparator))

    def append_frame_dict(self, dict):
        frame_dict = dict.copy()
        self.frames.append(frame_dict)

    def append_frame(self, frame_idx, img_file_name, Tcw, camera_intrinsic, frame_dim, time_stamp=0.0, depth_file_name=None, tag=None, rgb_intrinsic=None):
        """
        Add Frame to the collection
        :param frame_idx: frame index
        :param img_file_name: corresponding image name
        :param Tcw: the camera extrinsic matrix from world to camera coordinates, np.array
        :param camera_intrinsic: camera intrinsic information (fx, fy, cx, cy, k1, k2), np.array
        :param frame_dim: frame dimension (h, w)
        :param time_stamp: time stamp for current frame
        """
        if time_stamp is None:
            return

        frame_dict = {}
        frame_dict['file_name'] = img_file_name
        frame_dict['id'] = int(frame_idx)
        frame_dict['extrinsic_Tcw'] = Tcw
        frame_dict['camera_intrinsic'] = camera_intrinsic
        frame_dict['timestamp'] = time_stamp
        frame_dict['frame_dim'] = (int(frame_dim[0]), int(frame_dim[1]))
        frame_dict['tag'] = tag
        if depth_file_name is not None:
            frame_dict['depth_file_name'] = depth_file_name
        if rgb_intrinsic is not None:
            frame_dict['rgb_intrinsic'] = rgb_intrinsic

        self.frames.append(frame_dict)

    def dump_to_json(self, file_path):
        """
        Save the frame collection to json file
        :param file_path: output json file path
        """

        # Sort the frame by index
        self.sort_by_frame_idx()

        frame_instances = copy.deepcopy(self.frames)
        for frame in frame_instances:
            frame['extrinsic_Tcw'] = frame['extrinsic_Tcw'].ravel().tolist()
            frame['camera_intrinsic'] = frame['camera_intrinsic'].ravel().tolist()
            if 'rgb_intrinsic' in frame:
                frame['rgb_intrinsic'] = frame['rgb_intrinsic'].ravel().tolist()
        with open(file_path, 'w') as out_json_file:
            json.dump(frame_instances, out_json_file, indent=2)

    def load_json(self, json_file_path):
        """
        Load the frames from json file
        :rtype:
        :param json_file_path: Frame instance json file path
        """
        with open(json_file_path, 'r') as json_file:
            json_instance = json.load(json_file)
            self.frames = json_instance
            for frame in self.frames:
                frame['id'] = int(frame['id'])
                Tcw = np.asarray(frame['extrinsic_Tcw'], dtype=np.float32).ravel()
                if Tcw.shape[0] == 16:
                    Tcw = Tcw.reshape((4, 4))
                elif Tcw.shape[0] == 12:
                    Tcw = Tcw.reshape((3, 4))
                else:
                    raise Exception("Tcw dim should be either 3x4 or 4x4")
                frame['extrinsic_Tcw'] = Tcw

                if 'camera_intrinsic' in frame:
                    frame['camera_intrinsic'] = np.asarray(frame['camera_intrinsic'], dtype=np.float32)
                if 'frame_dim' in frame:
                    frame['frame_dim'] = (int(frame['frame_dim'][0]), int(frame['frame_dim'][1]))
                if 'rgb_intrinsic' in frame:
                    frame['rgb_intrinsic'] = np.asarray(frame['rgb_intrinsic'], dtype=np.float32)

        # Sort the frame by index
        self.sort_by_frame_idx()

    def get_frame_by_timestamp(self, timestamp):
        # Find the closest frame with timestamp
        if self.timestamp_cache is None:
            self.__build_timestamp_cache__()

        if timestamp > self.timestamp_cache[-1] or timestamp < self.timestamp_cache[0]:
            raise Exception("No Frame founded within overlap range.")

        target_idx = np.argmin(np.abs(self.timestamp_cache - timestamp))
        return self.frames[target_idx]

    def get_frame(self, idx):
        if idx < len(self.frames):
            return self.frames[idx]
        else:
            return None

    def get_image_name(self, frame_dict):
        if 'file_name' in frame_dict:
            return frame_dict['file_name']
        else:
            return None

    def get_depth_name(self, frame_dict):
        if 'depth_file_name' in frame_dict:
            return frame_dict['depth_file_name']
        else:
            return None

    def get_Tcw(self, frame_dict):
        if 'extrinsic_Tcw' in frame_dict:
            return frame_dict['extrinsic_Tcw']
        else:
            return None

    def get_Twc(self, frame_dict):
        if 'extrinsic_Tcw' in frame_dict:
            Tcw = frame_dict['extrinsic_Tcw']
            Twc = cam_opt.camera_pose_inv(Tcw[:3, :3], Tcw[:3, 3])
            return Twc
        else:
            return None

    def get_intrinsic(self, frame_dict):
        if 'camera_intrinsic' in frame_dict:
            return frame_dict['camera_intrinsic'][:4]
        else:
            return None

    def get_undist_params(self, frame_dict):
        if 'camera_intrinsic' in frame_dict:
            return frame_dict['camera_intrinsic'][4:]
        else:
            return None

    def get_K_mat(self, frame_dict):
        if 'camera_intrinsic' in frame_dict:
            return K_from_frame(frame_dict)
        else:
            return None

    def get_frame_dim(self, frame_dict):
        if 'frame_dim' in frame_dict:
            return frame_dict['frame_dim']
        else:
            return None

    def get_idx_n_timestamp(self, frame_dict):
        if 'timestamp' in frame_dict:
            return frame_dict['id'], frame_dict['timestamp']
        else:
            return None

    def get_tag(self, frame_dict):
        if 'tag' in frame_dict:
            return frame_dict['tag']
        else:
            return None
