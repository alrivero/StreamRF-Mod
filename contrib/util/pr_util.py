import torch
import numpy as np
import pandas as pd
import plotly.express as px
from pytorch3d.ops import knn_points

# Point cloud dialation based on 
# https://www.sciencedirect.com/science/article/pii/S0924271620302264?via%3Dihub
def dialate_pc(point_cloud, struct_elm, ref_point, max_threshold=0.15):
    N, _ = point_cloud.shape

    # 1. Compute the distance point-to-point (dpp)
    closest_5 = knn_points(point_cloud[None], point_cloud[None], K=5, norm=2, return_nn=True)[2].squeeze(0)
    dpp = (closest_5 - point_cloud.unsqueeze(1)).reshape(-1, 3).norm(dim=1).mean()

    # 2. (Slightly modified) Compute all translated structural elements and combine
    se_translation = point_cloud - ref_point
    all_struct_elm = struct_elm[None].expand(N, -1, -1) + se_translation.reshape(N, 1, -1)
    all_struct_elm = all_struct_elm.reshape(-1, 3)
    combined = torch.cat((all_struct_elm, point_cloud))

    # 3. Compute the closest point to all translated structural elements and those needed
    closest_se = knn_points(all_struct_elm[None], combined[None], K=2, norm=2, return_nn=True)[2][0, :, 1, :]
    needed_se = (all_struct_elm - closest_se).norm(dim=1) >= (dpp * 0.55)

    all_struct_elm = all_struct_elm[needed_se]

    rand_perm = torch.randperm(all_struct_elm.size(0))
    idx = rand_perm[:int(len(point_cloud) * max_threshold)]
    all_struct_elm = all_struct_elm[idx]

    origin_idx = (needed_se // struct_elm.shape[0])[idx]

    return all_struct_elm, origin_idx

# Used to visualize a point cloud as an interactive plot
def visualize_pc(point_clouds, titles, out_path, point_data):
    cloud_dfs = []
    for cloud, title, data in zip(point_clouds, titles, point_data):
        cloud_df = pd.DataFrame(cloud, columns = ['x','y','z'])
        if point_data is None:
            cloud_df["Category"] = title
        else:
            cloud_df["Category"] = point_data
        
        cloud_dfs.append(cloud_df)
    cloud_dfs = pd.concat(cloud_dfs).reset_index(drop=True)

    fig = px.scatter_3d(cloud_dfs, x='x', y='y', z='z', color="Category")
    fig.write_html(out_path)

