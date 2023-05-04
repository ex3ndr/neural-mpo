import torch


def cholesky_vector_to_matrix_t(vector, size):
    """
    Convert a packed Cholesky lower-triangular factorization vector into a Cholesky lower-triangular matrix.

    :param vector: A tensor of size (batch_size, n * (n + 1) // 2) containing packed Cholesky lower-triangular factorization.
    :param size: The size of the square Cholesky lower-triangular matrix.
    :return: A tensor of size (batch_size, size, size) containing the Cholesky lower-triangular matrices.
    """

    # Get the indices of the lower-triangular part of the matrix
    indices = torch.tril_indices(row=size, col=size, offset=0)

    # Create a zero tensor for the Cholesky lower-triangular matrices
    matrix = torch.zeros(size=(vector.size(0), size, size), dtype=torch.float32).to(vector.device)

    # Assign the elements of the input vector to the lower-triangular part of the Cholesky matrices
    matrix[:, indices[0], indices[1]] = vector

    return matrix


def cholesky_vector_size(size):
    return (size * (size + 1)) // 2
