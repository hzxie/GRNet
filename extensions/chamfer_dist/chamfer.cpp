#include <ATen/ATen.h>
#include <torch/extension.h>

int chamfer_forward(at::Tensor xyz1,
                    at::Tensor xyz2,
                    at::Tensor dist1,
                    at::Tensor dist2,
                    at::Tensor idx1,
                    at::Tensor idx2) {
  const int batch_size = xyz1.size(0);
  const int n          = xyz1.size(1);  // num_points point cloud A
  const int m          = xyz2.size(1);  // num_points point cloud B
  auto xyz1_accessor   = xyz1.accessor<float, 3>();
  auto xyz2_accessor   = xyz2.accessor<float, 3>();

  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < n; ++j) {
      float _min_distance(1e12);
      for (int k = 0; k < m; ++k) {
        float x1   = xyz1_accessor[i][j][0];
        float y1   = xyz1_accessor[i][j][1];
        float z1   = xyz1_accessor[i][j][2];
        float x2   = xyz2_accessor[i][k][0];
        float y2   = xyz2_accessor[i][k][1];
        float z2   = xyz2_accessor[i][k][2];
        float dx   = x1 - x2;
        float dy   = y1 - y2;
        float dz   = z1 - z2;
        float dist = dx * dx + dy * dy + dz * dz;
        if (dist < _min_distance) {
          _min_distance = dist;
          idx1[i][j]    = k;
        }
      }
      dist1[i][j] += _min_distance;
    }
  }
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < m; ++j) {
      float _min_distance(1e12);
      for (int k = 0; k < n; ++k) {
        float x1   = xyz2_accessor[i][j][0];
        float y1   = xyz2_accessor[i][j][1];
        float z1   = xyz2_accessor[i][j][2];
        float x2   = xyz1_accessor[i][k][0];
        float y2   = xyz1_accessor[i][k][1];
        float z2   = xyz1_accessor[i][k][2];
        float dx   = x1 - x2;
        float dy   = y1 - y2;
        float dz   = z1 - z2;
        float dist = dx * dx + dy * dy + dz * dz;
        if (dist < _min_distance) {
          _min_distance = dist;
          idx2[i][j]    = k;
        }
      }
      dist2[i][j] += _min_distance;
    }
  }

  return 1;
}

int chamfer_backward(at::Tensor xyz1,
                     at::Tensor xyz2,
                     at::Tensor gradxyz1,
                     at::Tensor gradxyz2,
                     at::Tensor grad_dist1,
                     at::Tensor grad_dist2,
                     at::Tensor idx1,
                     at::Tensor idx2) {
  const int batch_size     = xyz1.size(0);
  const int n              = xyz1.size(1);  // num_points point cloud A
  const int m              = xyz2.size(1);  // num_points point cloud B
  auto xyz1_accessor       = xyz1.accessor<float, 3>();
  auto xyz2_accessor       = xyz2.accessor<float, 3>();
  auto idx1_accessor       = idx1.accessor<int, 2>();
  auto idx2_accessor       = idx2.accessor<int, 2>();
  auto grad_dist1_accessor = grad_dist1.accessor<float, 2>();
  auto grad_dist2_accessor = grad_dist2.accessor<float, 2>();

  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < n; ++j) {
      int k      = idx1_accessor[i][j];
      float x1   = xyz1_accessor[i][j][0];
      float y1   = xyz1_accessor[i][j][1];
      float z1   = xyz1_accessor[i][j][2];
      float x2   = xyz2_accessor[i][k][0];
      float y2   = xyz2_accessor[i][k][1];
      float z2   = xyz2_accessor[i][k][2];
      float grad = grad_dist1_accessor[i][j] * 2;

      gradxyz1[i][j][0] += grad * (x1 - x2);
      gradxyz1[i][j][1] += grad * (y1 - y2);
      gradxyz1[i][j][2] += grad * (z1 - z2);
      gradxyz2[i][k][0] += grad * (x2 - x1);
      gradxyz2[i][k][1] += grad * (y2 - y1);
      gradxyz2[i][k][2] += grad * (z2 - z1);
    }
  }
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < m; ++j) {
      int k      = idx2_accessor[i][j];
      float x1   = xyz2_accessor[i][j][0];
      float y1   = xyz2_accessor[i][j][1];
      float z1   = xyz2_accessor[i][j][2];
      float x2   = xyz1_accessor[i][k][0];
      float y2   = xyz1_accessor[i][k][1];
      float z2   = xyz1_accessor[i][k][2];
      float grad = grad_dist2_accessor[i][j] * 2;

      gradxyz2[i][j][0] += grad * (x1 - x2);
      gradxyz2[i][j][1] += grad * (y1 - y2);
      gradxyz2[i][j][2] += grad * (z1 - z2);
      gradxyz1[i][k][0] += grad * (x2 - x1);
      gradxyz1[i][k][1] += grad * (y2 - y1);
      gradxyz1[i][k][2] += grad * (z2 - z1);
    }
  }

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &chamfer_forward, "Chamfer forward");
  m.def("backward", &chamfer_backward, "Chamfer backward");
}
