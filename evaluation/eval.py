import sys
sys.path.insert(0, "..")
from data.preprocess import points_in_boxes

import open3d as o3d
import numpy as np
from data.WaymoDataset import WaymoDataset
from data.util import ApplyPillarization, drop_points_function

from visualization.util import get_transfmat_by_points


from models.FastFlow3DModelScatter import FastFlow3DModelScatter
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils_eval import AugmentedPC, random_drop_points, drop_clusters, random_crop_points, predict_flows, \
    random_box_shadow_points, random_transform_points, open3d_visualize_points, open3d_visualize_several_points, \
    apply_tranformation, get_fasflow_error, get_gicp_error
import pygicp
import argparse

import yaml
from models.FastFlow3DModelScatter import FastFlow3DModelScatter
from data.util import custom_collate_batch
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/media/robot/hdd/waymo_open_dataset_scene_flow/train_prep", type=str)
parser.add_argument("--config", default="config.yaml", type=str)
parser.add_argument("--model_path", default="/home/robot/Downloads/last_.ckpt", type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--return_boxes", default=1, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = args.data_dir  #data_dir = "/media/robot/hdd/waymo_open_dataset_scene_flow/train_prep"
    waymo_dataset = WaymoDataset(data_dir)
    waymo_dataset.return_boxes = args.return_boxes

    # model = FastFlow3DModelScatter.load_from_checkpoint("/media/robot/hdd/repos/FastFlow3D/last_2nov.ckpt")
    model_path = args.model_path # "/home/robot/Downloads/last_.ckpt"
    model = FastFlow3DModelScatter.load_from_checkpoint(model_path).to(args.device)
    model.eval()
    config_info = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    # config_info = {"grid_cell_size": {"value": 0.33203125}, "x_min": {"value": -85}, "x_max": {"value": 85},
    #                "y_min": {"value": -85}, "y_max": {"value": 85}, "z_min": {"value": -3}, "z_max": {"value": 3},
    #                "n_pillars_x": {"value": 512}, "n_pillars_y": {"value": 512}}
    # save to yaml
    # with open("config.yaml", 'w') as outfile:
    #     yaml.dump(config_info, outfile, default_flow_style=False)

    grid_cell_size = config_info['grid_cell_size']['value']
    x_min = config_info['x_min']['value']
    y_min = config_info['y_min']['value']
    z_min = config_info['z_min']['value']
    x_max = config_info['x_max']['value']
    y_max = config_info['y_max']['value']
    z_max = config_info['z_max']['value']
    n_pillars_x = config_info['n_pillars_x']['value']
    n_pillars_y = config_info['n_pillars_y']['value']
    point_cloud_transform = ApplyPillarization(grid_cell_size=grid_cell_size, x_min=x_min,
                                               y_min=y_min, z_min=z_min, z_max=z_max, n_pillars_x=n_pillars_x)

    waymo_dataset.set_point_cloud_transform(point_cloud_transform)
    drop_points_function = drop_points_function(x_min=x_min,
                                                x_max=x_max, y_min=y_min, y_max=y_max,
                                                z_min=z_min, z_max=z_max)
    waymo_dataset.set_drop_invalid_point_function(drop_points_function)
    # augmenter = AugmentedPC(None, augment=[random_drop_points(0.2), drop_clusters(100),
    #                                random_crop_points(0.7), random_box_shadow_points(0.1)],
    #                 transform=None)
    augmenter = None

    errs_100 = {"stationar": [], "slow": [], "fast": []}
    errs_200 = {"stationar": [], "slow": [], "fast": []}
    errs_400 = {"stationar": [], "slow": [], "fast": []}
    errs_big = {"stationar": [], "slow": [], "fast": []}

    gicp_errs_100 = {"stationar": [], "slow": [], "fast": []}
    gicp_errs_200 = {"stationar": [], "slow": [], "fast": []}
    gicp_errs_400 = {"stationar": [], "slow": [], "fast": []}
    gicp_errs_big = {"stationar": [], "slow": [], "fast": []}

    vis_cur_iter = False
    pbar = tqdm(range(340, len(waymo_dataset)))
    for batch_number in pbar:
        waymo_dataset.pillarize(False)
        (previous_frame, current_frame), flows, current_bbox, previous_bbox, current_ind, previous_ind = waymo_dataset[batch_number]
        p_flow, masks, _ = predict_flows(model, waymo_dataset, batch_number, augmenter)
        current_frame = (current_frame[0][masks], None)
        for bn in range(len(current_bbox)):

            masks = points_in_boxes(current_bbox[bn][None], current_frame[0])
            num_points = masks.sum()
            if num_points > 20:
                try:
                    bnp = np.where(previous_ind == current_ind[bn])[0][0]
                except:
                    continue
                masks_prev = points_in_boxes(previous_bbox[bnp][None], previous_frame[0])
                tr_mat = get_transfmat_by_points(previous_bbox[bnp], current_bbox[bn])
                t_vec = tr_mat[:3, 3]
                speed = np.linalg.norm(t_vec*10)

                points_current = current_frame[0][masks]
                points_prev = previous_frame[0][masks_prev]
                if len(points_current) <10 or len(points_prev) < 10:
                    continue
                gicp_transform = pygicp.align_points(points_current[:,:3].astype("float64") - points_current.mean(axis=0)[:3],
                                             points_prev[:,:3].astype("float64") - points_current.mean(axis=0)[:3])

                points_prev_tr = apply_tranformation(points_prev[:,:3] - points_current.mean(axis=0)[:3], tr_mat) + points_current.mean(axis=0)[:3]
                # points_cur_tr = apply_tranformation(points_current[:, :3], tr_mat)
                # open3d_visualize_several_points([points_current, points_prev], bboxes=[current_bbox[bn], previous_bbox[bnp]])
                # open3d_visualize_several_points([points_current[:, :3], points_prev_tr] )
                # open3d_visualize_several_points([points_cur_tr[:, :3], points_prev])


                pm = p_flow[masks]
                # pm is in the same order as points_current and represents the flow from points_current to points_prev

                if vis_cur_iter:
                    pc_cur = o3d.geometry.PointCloud()
                    pc_cur.points = o3d.utility.Vector3dVector(points_current[:, :3])
                    pc_cur.paint_uniform_color([0.1, 0.1, 0.7])  # blue

                    pc_prev = o3d.geometry.PointCloud()
                    shift = np.array([3, 0, 0]) * 0
                    pc_prev.points = o3d.utility.Vector3dVector(points_prev[:, :3] + shift)
                    pc_prev.paint_uniform_color([0.7, 0.1, 0.1])  # red

                    pc_pred = o3d.geometry.PointCloud()
                    pc_pred.points = o3d.utility.Vector3dVector(points_current[:, :3] - pm / 10 + shift)
                    pc_pred.paint_uniform_color([0.7, 0.7, 0.1])

                    points_icp = apply_tranformation(points_current[:, :3] - points_current.mean(axis=0)[:3],
                                                         np.linalg.inv(gicp_transform)) + points_current.mean(axis=0)[:3]
                    pc_pred_icp = o3d.geometry.PointCloud()
                    pc_pred_icp.points = o3d.utility.Vector3dVector(points_icp[:, :3])
                    pc_pred_icp.paint_uniform_color([0.2, 0.2, 0.2])
                    # visualize
                    o3d.visualization.draw_geometries([pc_cur, pc_prev, pc_pred, pc_pred_icp])
                error_ff = get_fasflow_error(points_current[:, :3], points_prev[:, :3], pm / 10, tr_mat)
                error_gicp = get_gicp_error(points_current[:, :3], points_prev_tr, gicp_transform, tr_mat)
                if num_points < 100:
                    errs = errs_100
                    gicp_errs = gicp_errs_100
                if 100 <= num_points < 200:
                    errs = errs_200
                    gicp_errs = gicp_errs_200
                if 200 <= num_points < 400:
                    errs = errs_400
                    gicp_errs = gicp_errs_400
                if 400<=num_points:
                    errs = errs_big
                    gicp_errs = gicp_errs_big
                if speed < 0.2:
                    errs["stationar"].append(error_ff)
                    gicp_errs["stationar"].append(error_gicp)
                elif speed < 5:
                    errs["slow"].append(error_ff)
                    gicp_errs["slow"].append(error_gicp)
                else:
                    errs["fast"].append(error_ff)
                    gicp_errs["fast"].append(error_gicp)
                speed_stat = f"stat: {len(errs['stationar'])}, slow: {len(errs['slow'])}, fast: {len(errs['fast'])}"
                pbar.set_description(f"error: {error_ff:.3f}, speed: {speed:.3f}  {speed_stat}")


    def vis_errs(errs_100, errs_200, errs_400, errs_big, save_name="fastflow_mean_errors.png", model_name="FastFlow",
                 threshold=None):
        # mean error fastflow
        if threshold is None:
            mean_errors = np.array([[np.mean(errs_100["stationar"]), np.mean(errs_100["slow"]), np.mean(errs_100["fast"])],
                                    [np.mean(errs_200["stationar"]), np.mean(errs_200["slow"]), np.mean(errs_200["fast"])],
                                    [np.mean(errs_400["stationar"]), np.mean(errs_400["slow"]), np.mean(errs_400["fast"])],
                                    [np.mean(errs_big["stationar"]), np.mean(errs_big["slow"]), np.mean(errs_big["fast"])]])
        else:
            mean_errors = np.array([[(np.array(errs_100["stationar"])<threshold).mean(), (np.array(errs_100["slow"])<threshold).mean(), (np.array(errs_100["fast"])<threshold).mean()],
                                    [(np.array(errs_200["stationar"])<threshold).mean(), (np.array(errs_200["slow"])<threshold).mean(), (np.array(errs_200["fast"])<threshold).mean()],
                                    [(np.array(errs_400["stationar"])<threshold).mean(), (np.array(errs_400["slow"])<threshold).mean(), (np.array(errs_400["fast"])<threshold).mean()],
                                    [(np.array(errs_big["stationar"])<threshold).mean(), (np.array(errs_big["slow"])<threshold).mean(), (np.array(errs_big["fast"])<threshold).mean()]])

        # numbers for each category fastflow
        num_samples = np.array([[len(errs_100["stationar"]), len(errs_100["slow"]), len(errs_100["fast"])],
                                [len(errs_200["stationar"]), len(errs_200["slow"]), len(errs_200["fast"])],
                                [len(errs_400["stationar"]), len(errs_400["slow"]), len(errs_400["fast"])],
                                [len(errs_big["stationar"]), len(errs_big["slow"]), len(errs_big["fast"])]])
        # plot table with mean errors for different speed and number of points in the box
        plt.figure(figsize=(10, 10))
        plt.imshow(mean_errors, cmap="viridis")
        # add text to the table
        for i in range(mean_errors.shape[0]):
            for j in range(mean_errors.shape[1]):
                plt.text(j, i, f"{mean_errors[i, j]:.3f} \n ({num_samples[i, j]})", ha="center", va="center", color="w")
        plt.clim(0, 0.6)
        if threshold:
            plt.clim(0.5, 1)
            if threshold<0.2:
                plt.clim(0, 1)

        plt.colorbar()
        plt.xticks([0, 1, 2], ["stationar\n <0.2", "slow\n (0.2, 5)", "fast\n >5"])
        plt.yticks([0, 1, 2, 3], ["<100", "100-200", "200-400", ">400"])
        # add title
        if threshold is None:
            plt.title(f"{model_name} Mean error in meters (number os samples) for different speed and number of points in the box\n")
        else:
            plt.title(f"{model_name} accuracy (threshold: {threshold}) in meters (number os samples) for different speed and number of points in the box\n ")
        # add x and y labels
        plt.xlabel("Speed (m/s)")
        plt.ylabel("Number of points in the box")
        plt.savefig(save_name)
        plt.show()


    vis_errs(errs_100, errs_200, errs_400, errs_big, save_name="fastflow_mean_errors.png")
    vis_errs(gicp_errs_100, gicp_errs_200, gicp_errs_400, gicp_errs_big, save_name="gicp_mean_errors.png",
             model_name="GICP")

    vis_errs(errs_100, errs_200, errs_400, errs_big, save_name="fastflow_ac_01.png", threshold=0.1)
    vis_errs(gicp_errs_100, gicp_errs_200, gicp_errs_400, gicp_errs_big, save_name="gicp_ac_01.png",
             model_name="GICP", threshold=0.1)
    # threshold = 1
    vis_errs(errs_100, errs_200, errs_400, errs_big, save_name="fastflow_ac_1.png", threshold=0.5)
    vis_errs(gicp_errs_100, gicp_errs_200, gicp_errs_400, gicp_errs_big, save_name="gicp_ac_1.png",
                model_name="GICP", threshold=1)
    np.save("fastflow_errors.npy", [errs_100, errs_200, errs_400, errs_big])
    np.save("gicp_errors.npy", [gicp_errs_100, gicp_errs_200, gicp_errs_400, gicp_errs_big])
