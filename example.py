"""
Example usage
"""

import torch
from chamferdist import ChamferDistance


# Create two random pointclouds
# (Batchsize x Number of points x Number of dims)
pc1 = torch.randn(1, 100, 3).cuda().contiguous()
pc2 = torch.randn(1, 50, 3).cuda().contiguous()
pc1.requires_grad = True

# Initialize Chamfer distance module
chamferDist = ChamferDistance()
# Compute Chamfer distance, and indices of closest points
# - dist1 is direction pc1 -> pc2 (for each point in pc1,
#   gets closest point in pc2)
# - dist 2 is direction pc2 -> pc1 (for each point in pc2,
#   gets closest point in pc1)
dist1, dist2, idx1, idx2 = chamferDist(pc1, pc2)
print(dist1.shape, dist2.shape, idx1.shape, idx2.shape)

# Usually, dist1 is not equal to dist2 (as the associations
# vary, in both directions). To get a consistent measure,
# usually we average both the distances
cdist = 0.5 * (dist1.mean() + dist2.mean())
print('Chamfer distance:', cdist)
cdist.backward()
