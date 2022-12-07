import os

import numpy as np
import torch
from tqdm import tqdm

from data.util import custom_collate_batch
import scipy as sp


def flownet_batch(batch, model):
    previous = []
    current = []
    previous.append(batch[0][0][0][:, :, :3].to(model.device))  # Lidar properties are not needed
    previous.append(batch[0][0][1].to(model.device))
    previous.append(batch[0][0][2].to(model.device))
    current.append(batch[0][1][0][:, :, :3].to(model.device))  # Lidar properties are not needed
    current.append(batch[0][1][1].to(model.device))
    current.append(batch[0][1][2].to(model.device))
    flows = batch[1].to(model.device)
    new_batch = ((previous, current), flows)
    return new_batch

def predict_flows(model, dataset, offset, architecture="FastFlowNet"):
    if architecture == "FlowNet":
        dataset.pillarize(False)
        (previous_frame, current_frame), flows = dataset[offset]
        # We set batchsize of 1 for predictions
        batch = custom_collate_batch([((previous_frame, current_frame), flows)])
        batch = flownet_batch(batch, model)
        with torch.no_grad():
            output = model(batch[0])
        predicted_flows = output[0].data.cpu().numpy()
        return predicted_flows
    elif architecture == "FastFlowNet":  # This model always uses GPU
        dataset.pillarize(True)
        (previous_frame, current_frame), flows, _, _, _, _ = dataset[offset]
        # (previous_frame, current_frame), flows = dataset[offset]
        batch = custom_collate_batch([((previous_frame, current_frame), flows)])
        with torch.no_grad():
            output = model(batch[0])
        predicted_flows = output[0].data.cpu().numpy()
        return predicted_flows
    else:
        print(f"Architecture {architecture} not implemented")
        exit(1)


def get_flows(dataset, index):
    flows_folder = os.path.join(dataset.data_path, "flows")
    flows_name = os.path.join(flows_folder, "flows_" + dataset.get_name_current_frame(index))
    predicted_flows = np.load(flows_name)['flows']
    return predicted_flows


def flows_exist(dataset):
    flows_folder = os.path.join(dataset.data_path, "flows")
    check_existing_folder = os.path.isdir(flows_folder)

    # If folder doesn't exist, then create it.
    if not check_existing_folder:
        return False

    else:
        print(f"Flows already exist, please remove {flows_folder} to process again the flows")
        print("Using already predicted flows...")
        return True

def predict_and_store_flows(model, dataset, architecture):
    flows_folder = os.path.join(dataset.data_path, "flows")
    check_existing_folder = os.path.isdir(flows_folder)

    # If folder doesn't exist, then create it.
    if not check_existing_folder:
        os.makedirs(flows_folder)
        print("created folder : ", flows_folder)

    else:
        print(f"Flows already exist, please remove {flows_folder} to process again the flows")
        print("Using already predicted flows...")
        return

    for i in tqdm(range(0, len(dataset)), desc="Predicting flows..."):
        predicted_flows = predict_flows(model, dataset, i, architecture)
        flows_name = os.path.join(flows_folder, "flows_" + dataset.get_name_current_frame(i))
        np.savez_compressed(flows_name, flows=predicted_flows)


def get_transfmat(bbox1, bbox2):
    """Get the rotation matrix from bbox1 to bbox2.
    Args:
        bbox1 (np.ndarray): (7, ) [x, y, z, w, l, h, yaw]
        bbox2 (np.ndarray): (7, ) [x, y, z, w, l, h, yaw]

    Returns:
        np.ndarray: (4, 4) transf matrix
    """
    yaw1 = bbox1[6]
    yaw2 = bbox2[6]
    yaw_diff = yaw2 - yaw1
    yaw_diff = yaw_diff % (2 * np.pi)
    if yaw_diff > np.pi:
        yaw_diff -= 2 * np.pi
    delta = bbox2[:3] - bbox1[:3]
    rot_mat = np.array([[np.cos(yaw_diff), -np.sin(yaw_diff), 0],
                        [np.sin(yaw_diff), np.cos(yaw_diff), 0],
                        [0, 0, 1]])
    transf_mat = np.eye(4)
    transf_mat[:3, :3] = rot_mat
    transf_mat[:3, 3] = delta
    return transf_mat

def get_transfmat_by_points(bbox1, bbox2):
    """Get the rotation matrix from bbox1 to bbox2.
    Args:
        bbox1 (np.ndarray): (8, 3) - points of the bounding box
        bbox2 (np.ndarray): (8, 3) - points of the bounding box

    Returns:
        np.ndarray: (4, 4) transf matrix
    """
    # Get the center of the bounding box
    center1 = np.mean(bbox1, axis=0)
    center2 = np.mean(bbox2, axis=0)
    b1 = bbox1 - center1
    b2 = bbox2 - center2
    transf = sp.spatial.transform.Rotation.align_vectors(b1, b2)
    # Get the rotation matrix
    rot_mat = transf[0].as_matrix()
    #np.linalg.inv(np.cov(bbox1.T - center1[:,None])) @ np.cov(bbox2.T - center2[:,None])
    # Get the translation
    delta = center2 - center1
    transf_mat = np.eye(4)
    transf_mat[:3, :3] = rot_mat
    transf_mat[:3, 3] = delta
    return transf_mat
