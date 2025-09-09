import torch

nums = torch.arange(12).reshape(3, 4)
print(f"Tensor:\n{nums}")

# The third column
col = nums[:, 2]
print(f"\nThe second column (the 'first' is actually the 0th): {col}")

scores = torch.rand(3, 2, 4)
print(f"\nNew tensor:\n{scores}")

matrix = scores[:, 1, :]
print(f"\nThe matrix from fixing the 1st dimension of the tensor:\n{matrix}")

col = scores[:, 1, 2]
print(f"\nThe column from fixing the 1st and 2nd dimension of the tensor:\n{col}")