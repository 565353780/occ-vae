import sys

sys.path.append("../point-cept")

import torch

from pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata import (
    PointTransformerV3,
)

from triline_vae.Model.query_fusion import QueryFusion


def test():
    points = [
        torch.rand(120000, 3).cuda(),
        torch.rand(20000, 3).cuda(),
        torch.rand(30000, 3).cuda(),
        torch.rand(40000, 3).cuda(),
    ]

    coords = torch.cat(points, dim=0)

    batch_indices = [
        torch.full((t.shape[0],), i, dtype=torch.long) for i, t in enumerate(points)
    ]
    batch = torch.cat(batch_indices, dim=0).cuda()

    data = {
        "coord": coords,
        "feat": coords,
        "batch": batch,
        "grid_size": 0.01,
    }

    ptv3 = PointTransformerV3(3, enc_mode=True).cuda()

    query_fusion = QueryFusion(512, 64, 256).cuda()

    point = ptv3(data)

    batch = point.batch
    feature = point.feat

    query_feature = query_fusion(feature, batch)

    print("coords shape: ", coords.shape)
    print("batch shape: ", batch.shape)
    print("point feature shape: ", feature.shape)
    print("query feature shape: ", query_feature.shape)
    return True
