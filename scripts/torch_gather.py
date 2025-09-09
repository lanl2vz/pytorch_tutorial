import torch

# torch.gather(input, dim, index, *, sparse_grad=False, out=None)
# input and index must have the same number of dimensions. 
########### IMPORTANT ###########
# It is also required that index.size(d) <= input.size(d) for all dimensions d != dim.
# For d == dim, index can take any value
#################################
# out will have the same shape as index. 
# For each value in index, it picks the value in input along the dimension dim.
# out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
# out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
# out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
# Note that input and index do not broadcast against each other.

# 2-dim tensor gather example
tensor = torch.arange(16).reshape(4, 4)
print(f"Tensor:\n{tensor}")
indices_to_collect = torch.tensor([[2], [1], [3]])
gathered = torch.gather(tensor, dim=1, index=indices_to_collect)
print(f"\nGathered values from the 1st dimension of the tensor:\n{gathered}")
print(f"Gathered shape: {gathered.shape}")

gathered = torch.gather(tensor, dim=0, index=indices_to_collect)
print(f"\nGathered values from the 0th dimension of the tensor:\n{gathered}")
print(f"Gathered shape: {gathered.shape}")

# 3-dim tensor gather example
tensor = torch.rand(3, 2, 4)
print(f"\n\nOriginal shape: {tensor.shape}")
print(f"\nTensor:\n{tensor}")

indices_to_collect = torch.tensor([[[0],[1]], [[1],[1]], [[1],[1]]])
gathered = torch.gather(tensor, dim=1, index=indices_to_collect)
print(f"\nGathered values from the 1st dimension of the tensor:\n{gathered}")
print(f"Gathered shape: {gathered.shape}")

indices_to_collect = torch.tensor([[[0]], [[1]], [[1]], [[1]]])
gathered = torch.gather(tensor, dim=0, index=indices_to_collect)
print(f"\nGathered values from the 0th dimension of the tensor:\n{gathered}")
print(f"Gathered shape: {gathered.shape}")

indices_to_collect = torch.tensor([[[0]]])
gathered = torch.gather(tensor, dim=0, index=indices_to_collect)
print(f"\nGathered values from the 0th dimension of the tensor:\n{gathered}")
print(f"Gathered shape: {gathered.shape}")