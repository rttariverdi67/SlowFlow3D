"""
MIT License

Copyright (c) 2021 Felix (Jabb0), Aron (arndz), Carlos (cmaranes)
Source: https://github.com/Jabb0/FastFlow3D

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from data.util import get_coordinates_and_features, get_bbox


# TODO: tensor operations to make it faster?
# TODO: check context name to ensure two consecutive frames -> a session should be consecutive
class WaymoDataset(Dataset):
    """
    Waymo Custom Dataset for flow estimation. For a detailed description of each
    field please refer to:
    https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto
    """

    def __init__(self, data_path,
                 drop_invalid_point_function=None,
                 point_cloud_transform=None,
                 n_points=None,
                 apply_pillarization=True,
                 return_boxes = False):
        """
        Args:
            data_path (string): Folder with the compressed data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            n_points (int): Number of maximum points. If None all points will be used
        """
        super().__init__()
        # Config parameters
        metadata_path = os.path.join(data_path, 'metadata')
        # It has information regarding the files and transformations

        self.data_path = data_path
        self.return_boxes = return_boxes
        self._drop_invalid_point_function = drop_invalid_point_function
        self._point_cloud_transform = point_cloud_transform

        # This parameter is useful when visualizing, since we need to pass
        # the pillarized point cloud to the model for infer but we would
        # like to display the points without pillarizing them
        self._apply_pillarization = apply_pillarization

        try:
            with open(metadata_path, 'rb') as metadata_file:
                self.metadata = pickle.load(metadata_file)
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found, please create it by running preprocess.py")

        self._n_points = n_points

    def __len__(self) -> int:
        return len(self.metadata['look_up_table'])

    def __getitem__(self, index):
        """
        Return two point clouds, the current point and its previous one. It also
        return the flow per each point of the current cloud

        A point cloud has a shape of [N, F], being N the number of points and the
        F to the number of features, which is [x, y, z, intensity, elongation]
        """
        current_frame, previous_frame, current_bbox, previous_bbox, current_ind, previous_ind, current_type_dict, previous_type_dict = self.read_point_cloud_pair(index)
        current_frame_pose, previous_frame_pose = self.get_pose_transform(index)
        flows = self.get_flows(current_frame)

        if self._n_points is not None:
            current_frame, previous_frame, flows = self.subsample_points(current_frame, previous_frame, flows)

        # G_T_C -> Global_TransformMatrix_Current
        G_T_C = np.reshape(np.array(current_frame_pose), [4, 4])

        # G_T_P -> Global_TransformMatrix_Previous
        G_T_P = np.reshape(np.array(previous_frame_pose), [4, 4])
        C_T_P = np.linalg.inv(G_T_C) @ G_T_P
        # https://github.com/waymo-research/waymo-open-dataset/blob/bbcd77fc503622a292f0928bfa455f190ca5946e/waymo_open_dataset/utils/box_utils.py#L179
        previous_frame = get_coordinates_and_features(previous_frame, transform=C_T_P)
        current_frame = get_coordinates_and_features(current_frame, transform=None)
        if previous_bbox is not None:
            previous_bbox = get_bbox(previous_bbox, transform=C_T_P)

        # Drop invalid points according to the method supplied
        if self._drop_invalid_point_function is not None:
            current_frame, flows = self._drop_invalid_point_function(current_frame, flows)
            previous_frame, _ = self._drop_invalid_point_function(previous_frame, None)

        # Perform the pillarization of the point_cloud
        if self._point_cloud_transform is not None and self._apply_pillarization:
            current_frame = self._point_cloud_transform(current_frame)
            previous_frame = self._point_cloud_transform(previous_frame)
        else:
            # output must be a tuple
            previous_frame = (previous_frame, None)
            current_frame = (current_frame, None)
        # This returns a tuple of augmented pointcloud and grid indices
        if self.return_boxes:
            return (previous_frame, current_frame), flows, current_bbox, previous_bbox, current_ind, previous_ind, current_type_dict, previous_type_dict
        return (previous_frame, current_frame), flows

    def subsample_points(self, current_frame, previous_frame, flows):
        # current_frame.shape[0] == flows.shape[0]
        if current_frame.shape[0] > self._n_points:
            indexes_current_frame = np.linspace(0, current_frame.shape[0]-1, num=self._n_points).astype(int)
            current_frame = current_frame[indexes_current_frame, :]
            flows = flows[indexes_current_frame, :]
        if previous_frame.shape[0] > self._n_points:
            indexes_previous_frame = np.linspace(0, previous_frame.shape[0]-1, num=self._n_points).astype(int)
            previous_frame = previous_frame[indexes_previous_frame, :]
        return current_frame, previous_frame, flows


    def pillarize(self, apply_pillarization):
        self._apply_pillarization = apply_pillarization

    def get_flow_ranges(self):
        flows_information = self.metadata['flows_information']
        mins = [flows_information['min_vx'], flows_information['min_vy'], flows_information['min_vz']]
        maxs = [flows_information['max_vx'], flows_information['max_vy'], flows_information['max_vz']]
        return np.array(mins), np.array(maxs)

    def set_drop_invalid_point_function(self, drop_invalid_point_function):
        self._drop_invalid_point_function = drop_invalid_point_function

    def set_point_cloud_transform(self, point_cloud_transform):
        self._point_cloud_transform = point_cloud_transform

    def set_n_points(self, n_points):
        self._n_points = n_points

    def get_name_current_frame(self, index):
        if index >= len(self):
            print("Index is out of the length of the dataset")
            exit(1)
        return self.metadata['look_up_table'][index][0][0]

    def read_point_cloud_pair(self, index):
        """
        Read from disk the current and previous point cloud given an index
        """
        # In the lookup table entries with (current_frame, previous_frame) are stored
        #print(self.metadata['look_up_table'][index][0][0])
        current = np.load(os.path.join(self.data_path, self.metadata['look_up_table'][index][0][0]), allow_pickle=True)
        current_frame = current['frame']
        previous = np.load(os.path.join(self.data_path, self.metadata['look_up_table'][index][1][0]), allow_pickle=True)
        previous_frame = previous['frame']
        current_bboxes, previous_bboxes = None, None
        current_ids, previous_ids = None, None

        current_type_dict = dict(enumerate(current['type_dict'].flatten(), 0))[0]
        previous_type_dict = dict(enumerate(previous['type_dict'].flatten(), 0))[0]

        if self.return_boxes:
            current_bboxes = current['bboxes']
            previous_bboxes = previous['bboxes']
            current_ids = current['obj_ids']
            previous_ids = previous['obj_ids']

        return current_frame, previous_frame, current_bboxes, previous_bboxes, current_ids, previous_ids, current_type_dict, previous_type_dict
        # return current_frame, previous_frame

    def get_pose_transform(self, index):
        """
        Return the frame poses of the current and previous point clouds given an index
        """
        current_frame_pose = self.metadata['look_up_table'][index][0][1]
        previous_frame_pose = self.metadata['look_up_table'][index][1][1]
        return current_frame_pose, previous_frame_pose

    def get_flows(self, frame):
        """
        Return the flows given a point cloud
        """
        flows = frame[:, -4:]
        return flows
