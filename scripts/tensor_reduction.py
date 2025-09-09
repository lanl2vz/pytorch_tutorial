import torch

# Mean reduction on different dimensions

scores = torch.rand( 3, 2, 4)
print(f"Original shape: {scores.shape}")
print(f"Tensor:\n{scores}")
reduction_0 = torch.mean(scores, dim=0)
print(f"Tensor reduction on the 0th dimension:\n{reduction_0}")
print(f"Reduced shape: {reduction_0.shape}")


scores = torch.rand( 3, 2, 4)
print(f"Original shape: {scores.shape}")
print(f"Tensor:\n{scores}")
reduction_1 = torch.mean(scores, dim=1)
print(f"Tensor reduction on the 1st dimension:\n{reduction_1}")
print(f"Reduced shape: {reduction_1.shape}")

# Argmax reduction on different dimensions
scores = torch.rand( 3, 2, 4)
print(f"Original shape: {scores.shape}")
print(f"Tensor:\n{scores}")
reduction_0 = torch.argmax(scores, dim=0)
print(f"Tensor reduction on the 0th dimension:\n{reduction_0}")
print(f"Reduced shape: {reduction_0.shape}")


scores = torch.rand( 3, 2, 4)
print(f"\n\nOriginal shape: {scores.shape}")
print(f"Tensor:\n{scores}")
reduction_1 = torch.argmax(scores, dim=1)
print(f"Tensor reduction on the 1st dimension:\n{reduction_1}")
print(f"Reduced shape: {reduction_1.shape}")