from data.preprocess import points_in_boxes
import sys
sys.path.append("/media/robot/hdd/repos/FastFlow3D")
import yaml
import open3d as o3d
import numpy as np
from data.WaymoDataset import WaymoDataset
from data.util import ApplyPillarization, drop_points_function

from visualization.util import predict_and_store_flows, flows_exist, get_transfmat_by_points
from models.FastFlow3DModelScatter import FastFlow3DModelScatter
import sys
sys.path.insert(0, "/media/robot/hdd/repos/FastFlow3D")
from models.FastFlow3DModelScatter import FastFlow3DModelScatter
from data.util import custom_collate_batch
import torch
#clustering from scikit-learn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def open3d_visualize_several_points(list_points, bboxes=None, color=None):
    to_vis = []
    # draw axis:
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=list_points[0][:,:3].mean(axis=0) + np.array([0,0,5]))
    # to_vis.append(axis)
    for points in list_points:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        if color is not None:
            pcd.colors = o3d.utility.Vector3dVector(color)
        else:
            # equal color for each all points
            pcd.colors = o3d.utility.Vector3dVector(np.ones_like(points[:, :3]) * np.random.rand(3))
        to_vis.append(pcd)

    if bboxes is not None:
        lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                 [4, 5], [5, 6], [6, 7], [4, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        for corner_box in bboxes:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corner_box)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            to_vis.append(line_set)
    o3d.visualization.draw_geometries(to_vis)

def open3d_visualize_points(points, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd])

class AugmentedPC:
    def __init__(self, config, augment, transform=None):
        self.augment = augment
        self.transform = transform
        self.config = config

    def __call__(self, point_cloud):
        tm = np.eye(4)
        mask = np.ones_like(point_cloud[:, 0], dtype=bool)
        if self.augment:
            choose = np.random.choice(len(self.augment))
            point_cloud, mask = self.augment[choose](point_cloud)
        if self.transform is not None:
            point_cloud, tm = self.transform(point_cloud)

        return point_cloud, tm, mask



def random_drop_points(drop_percentage=0.5):
    def rand_drop_points_function(points):
        masks = np.random.rand(points.shape[0]) > drop_percentage
        points = points[masks]
        return points, masks
    return rand_drop_points_function

def drop_clusters(n_clusters=10):
    def drop_clusters_function(points):
        # cluster points
        z = points[:, 0]+0
        z.sort(axis=0)
        z = z[int(len(z)*0.01): int(len(z)*0.99)]
        criterias = np.linspace(z.min(), z.max(), n_clusters)
        labels = np.digitize(points[:,0], criterias)
        # choose 20 labels to be masked
        labels_to_mask = np.random.choice(np.unique(labels), 50)
        masks = ~np.isin(labels, labels_to_mask)
        points = points[masks]
        # db = KMeans(n_clusters=n_clusters).fit(points[:,:3]) #? only z
        # labels = db.labels_
        # # get cluster sizes
        # cluster_sizes = np.zeros(db.n_clusters)
        # for i in range(db.n_clusters):
        #     cluster_sizes[i] = np.sum(labels == i)
        # # get cluster with most points
        # median_cluster = np.argsort(cluster_sizes)[len(cluster_sizes) // 2]
        #
        # masks = labels != median_cluster
        # points = points[masks]

        return points, masks
    return drop_clusters_function

def random_crop_points(crop_part=0.1):
    def crop_points_function(points):
        # get random point
        random_point = points[np.random.randint(points.shape[0])]

        # crop size is size of points * crop_part
        crop_size = np.array([points[:,0].max() - points[:,0].min(),
                              points[:,1].max() - points[:,1].min(),
                                points[:,2].max() - points[:,2].min()])/2 * crop_part # + points[:,0:3].mean(axis=0)
        # get points in a cube around the random point
        masks = (points[:,0] < (random_point[0] + crop_size[0])) & (points[:,0] > (random_point[0] - crop_size[0])) & \
                (points[:,1] < (random_point[1] + crop_size[1])) & (points[:,1] > (random_point[1] - crop_size[1])) & \
                (points[:,2] < (random_point[2] + crop_size[2])) & (points[:,2] > (random_point[2] - crop_size[2]))

        points = points[masks]
        return points, masks
    return crop_points_function

def random_box_shadow_points(crop_part=0.1):
    def random_box_shadow_points_function(points):
        # get random point
        random_point = points[np.random.randint(points.shape[0])]

        # crop size is size of points * crop_part
        crop_size = np.array([points[:,0].max() - points[:,0].min(),
                              points[:,1].max() - points[:,1].min(),
                                points[:,2].max() - points[:,2].min()])/2 * crop_part # + points[:,0:3].mean(axis=0)
        # get points in a cube around the random point
        masks = (points[:,0] < (random_point[0] + crop_size[0])) & (points[:,0] > (random_point[0] - crop_size[0])) & \
                (points[:,1] < (random_point[1] + crop_size[1])) & (points[:,1] > (random_point[1] - crop_size[1])) & \
                (points[:,2] < (random_point[2] + crop_size[2])) & (points[:,2] > (random_point[2] - crop_size[2]))

        points = points[~masks]
        return points, masks
    return random_box_shadow_points_function
# test drop_clusters_function


def plot_points(points, color=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color)
    plt.show()


def random_transform_points(max_angle_deg=10, max_translation=0.5):
    def random_transform_points_function(points):
        # get random angle
        angle = np.random.rand() * 2 * np.pi
        angle_max = max_angle_deg / 180 * np.pi
        angle = np.clip(angle, -angle_max, angle_max)
        # ger random translation
        translation = np.random.rand(3) * 0.2 - 0.1
        # get rotation matrix
        rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1]])
        # get transformation matrix
        tm = np.eye(4)
        tm[0:3, 0:3] = rot_mat
        tm[0:3, 3] = translation
        # transform points
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        points = np.dot(points, tm.T)
        points = points[:, 0:3]
        return points, tm
    return random_transform_points_function



def predict_flows(model, dataset, offset, augmenter=None):
        dataset.pillarize(True)
        (previous_frame, current_frame), flows, _, _, _, _ = dataset[offset]
        # (previous_frame, current_frame), flows = dataset[offset]
        mask = np.ones_like(current_frame[0][:, 0], dtype=np.bool)
        transform = np.eye(4)
        if augmenter is not None:
            current_frame_points, tranform, mask = augmenter(current_frame[0])
            current_frame = (current_frame_points, current_frame[1][mask])
            flows = flows[mask]
        batch = custom_collate_batch([((previous_frame, current_frame), flows)])
        # batch -> ((tensor, tensor, tensor), (tensor, tensor, tensor)), tensor
        # move all tensors to device
        device = model.device
        batch = ((batch[0][0][0].to(device), batch[0][0][1].to(device), batch[0][0][2].to(device)),
                (batch[0][1][0].to(device), batch[0][1][1].to(device), batch[0][1][2].to(device)))
        with torch.no_grad():
            output = model(batch)
        predicted_flows = output[0].data.cpu().numpy()
        return predicted_flows, mask, transform


def apply_tranformation(points: np.ndarray, transform: np.ndarray):
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = np.dot(transform, points.T).T
    points = points[:, 0:3]
    return points


def get_fasflow_error(points_current: np.array, points_prev: np.array, pm: np.array, tr_mat: np.array):
    """
    Parameters
    ----------
    points_current - points in current frame shape (N, 3)
    points_prev - points in previous frame shape (N1, 3)
    pm - predicted flow shape (N, 3)
    tr_mat - transformation matrix from previous to current

    Returns
    -------
    error - mean error in meters
    """
    # apply transformation to points
    center = points_current.mean(axis=0)
    points_prev_corrected = apply_tranformation(points_current - center, np.linalg.inv(tr_mat)) + center
    predicted_prev = points_current - pm
    # calculate error
    error = np.mean(np.linalg.norm(points_prev_corrected - predicted_prev, axis=1))
    return error

def get_gicp_error(points_current: np.array, points_prev_tr: np.array, gicp_transform: np.array, gt_mat: np.array):

    """
    Parameters
    ----------
    points_current - points in current frame shape (N, 3)
    points_prev - points in previous frame shape (N1, 3)
    gicp_transform - predicted transform (4, 4) (from previous to current)
    gt_mat - transformation matrix from previous to current (4, 4)

    Returns
    -------
    error - mean error in meters
    """
    # 1. apply gt transformation to current points - mean current points
    # 2. apply pred transformation to current points
    # 3. calculate error
    center = points_current.mean(axis=0)
    points_prev_corrected_gt = apply_tranformation(points_current - center, np.linalg.inv(gt_mat)) + center
    points_prev_corrected_pred = apply_tranformation(points_current - center, np.linalg.inv(gicp_transform)) + center
    error = np.mean(np.linalg.norm(points_prev_corrected_gt - points_prev_corrected_pred, axis=1))
    return error


# main
if __name__ == '__main__':
    augmenter = AugmentedPC(None, augment=[random_drop_points(0.5), drop_clusters(0.1),
                               random_crop_points(0.7), random_box_shadow_points(0.1)],
                transform=random_transform_points(10, 0.5))

    points = np.random.rand(1000, 3)
    for i in range(10):
        a_p, tm = augmenter(points)
        plot_points(a_p)
    # clustered = drop_clusters(0.1)(points)
    # plot_points(clustered)
    # droped = random_drop_points(0.8)(points)
    # plot_points(droped)
    # cropped = random_crop_points(0.5)(points)
    # plot_points(cropped)
    # shad = random_box_shadow_points(0.6)(points)
    # plot_points(shad)
    # tr_points, tm = random_transform_points()(points)
    # plot_points(tr_points)
    # load config
    # config = load_config()
    # # load dataset
    # dataset = KittiDataset(config, augment=[random_drop_points(0.5), drop_clusters(0.1), random_crop_points(0.1), random_box_shadow_points(0.1)], transform=random_transform_points(10, 0.5))
    # # create model
    # model = FlowNetS(config)
    # # load model
    # model.load_state_dict(torch.load('FlowNetS_checkpoint.pth.tar'))
    # # set model to eval mode
    # model.eval()
    # # predict flows
    # predicted_flows = predict_flows(model, dataset, 0)
    # # plot flows
    # plot_points(predicted_flows, color='red')
    # plot_points(dataset[0][1], color='blue')
