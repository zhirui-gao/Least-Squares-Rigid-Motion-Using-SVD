import numpy as np
def rigid_transform_2d( A, B, weights=None, weight_threshold=0):
    """
    Input:
        - A:       [bs, num_corr, 2], source point cloud
        - B:       [bs, num_corr, 2], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0

     # find mean of point cloud
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)
    H = Am.permute(0, 2, 1) @ Weight @ Bm
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)


    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(2)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0,2,1) - R @ centroid_A.permute(0,2,1)
    print('Estimated R:\n',R)
    print('Estimated T:\n',t)

bs_src_point = np.array([[[1.,0],[2, 1],[3, 2],[1,1]]]) #
bs_src_point = bs_src_point[0].T
bs_src_point = bs_src_point[None]
theta = 30
theta = (theta/180.0)*np.pi
real_rot = np.array([ [np.cos(theta),-np.sin(theta)],
                      [np.sin(theta),np.cos(theta)]
])
real_t = np.array([10,20]).reshape(2,1)
bs_tgt_point = real_rot[None,:,:]@data + real_t[None]


src_keypts = torch.tensor(bs_src_point).permute(0,2,1).to(torch.float32)
tgt_keypts = torch.tensor(bs_tgt_point).permute(0,2,1).to(torch.float32)

print('gt R:\n',real_rot)
print('gt T:\n',real_t)
rigid_transform_2d(src_keypts,tgt_keypts)
