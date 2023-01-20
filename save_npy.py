import numpy as np
from data.WaymoDataset import WaymoDataset

import os
from data.preprocess import points_in_boxes
import argparse
from tqdm import tqdm

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
    # Get the rotation matrix
    rot_mat = np.linalg.inv(np.cov(bbox1.T)) @ np.cov(bbox2.T)
    # Get the translation
    delta = center2 - center1
    transf_mat = np.eye(4)
    transf_mat[:3, :3] = rot_mat
    transf_mat[:3, 3] = delta
    return transf_mat



def save_file(name, data):
    save_to = os.path.dirname(name)
    os.makedirs(save_to, exist_ok=True)
    np.save(name, data)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_path", type=str, default='../patata/datasets/waymo_flow/raw_data_directory/')
parser.add_argument("--subset", type=str)
parser.add_argument("--split", type=int, default=1)
parser.add_argument("--chunk_num", type=int, default=0)



if __name__=='__main__':
    args = parser.parse_args()
    data_dir = os.path.join(args.data_path, "metadata", args.subset)

    waymo_dataset = WaymoDataset(data_dir)
    waymo_dataset.return_boxes = True



    IDs = [i for i in range(len(waymo_dataset))]
    cunks = np.array_split(IDs, args.split)
    start_id = cunks[args.chunk_num][0]
    end_id = cunks[args.chunk_num][-1]




    for idx_dataset in tqdm(range(start_id, end_id)):
        try:
            (previous_frame, current_frame), flows, c_bbox, p_bbox, c_ids, p_ids, current_type_dict, previous_type_dict = waymo_dataset[idx_dataset]
            for counter, id_ in enumerate(c_ids):
                # get current PC inside given bbox[id]

                name = f"{idx_dataset:09d}_{counter:09d}"
                cp = current_frame[0][:, :3][points_in_boxes(c_bbox[counter][None], current_frame[0])]
                if id_ not in p_ids:
                    continue
                if cp.shape[0] < 300:
                    continue

                data_TYPEs = {
                    1:'TYPE_VEHICLE',
                    2:'TYPE_PEDESTRIAN',
                    3:'TYPE_SIGN'}

                if current_type_dict[id_] not in list(data_TYPEs.keys()):
                    continue

                data_TYPE = data_TYPEs[current_type_dict[id_]]

                
                # get previous PC with the same bbox id
                cpp = previous_frame[0][:, :3][points_in_boxes(p_bbox[(id_ == p_ids).argmax()][None], previous_frame[0])]
                
                if cpp.shape[0] < 300:
                
                    continue
                # get GT transf max from bbox
                tr_mat = get_transfmat_by_points(c_bbox[counter], p_bbox[(id_ == p_ids).argmax()])

                name = os.path.join(args.data_path, 'waymo_npy_updated',args.subset ,'current', data_TYPE, name)

                save_file(name, cp)
                save_file(name.replace('current', 'previous'), cpp)
                save_file(name.replace('current', 'gt_poses'), tr_mat)
                

        except IndexError: 
            pass








# conda activate slow
# python save_npy.py --subset val --chunk_num 0
