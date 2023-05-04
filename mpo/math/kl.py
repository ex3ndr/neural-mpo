import torch


def categorical_kl(p1, p2):
    """
    Calculate the Kullback-Leibler (KL) divergence between two Categorical distributions.

    :param p1: A tensor of size (B, D) representing a batch of categorical distributions.
    :param p2: A tensor of size (B, D) representing another batch of categorical distributions.
    :return: The KL divergence between the two input distributions.
    """
    # Clamp both distributions to avoid zero division
    p1 = torch.clamp_min(p1, 1e-6)  # Not needed, but to make it even more numerically stable
    p2 = torch.clamp_min(p2, 1e-6)

    # Calculate the KL divergence between the two distributions
    return torch.mean((p1 * torch.log(p1 / p2)).sum(dim=-1))


def gaussian_kl(mu_i, mu, a_i, a):
    """
    Calculate the decoupled KL divergence between two multivariate Gaussian distributions.

    :param mu_i: A tensor of size (B, n) representing the means of the first batch of Gaussian distributions.
    :param mu: A tensor of size (B, n) representing the means of the second batch of Gaussian distributions.
    :param a_i: A tensor of size (B, n, n) representing the lower-triangular Cholesky factorization of the covariances
               of the first batch of Gaussian distributions.
    :param a: A tensor of size (B, n, n) representing the lower-triangular Cholesky factorization of the covariances
              of the second batch of Gaussian distributions.
    :return: C_μ, C_Σ, mean_det_Σi, mean_det_Σ: Scalars representing the mean and covariance terms of the KL divergence,
             and the mean determinants of the covariance matrices Σi and Σ.
    """

    def batch_transpose(m):
        return m.transpose(dim0=-2, dim1=-1)

    def batch_trace(m):
        return m.diagonal(dim1=-2, dim2=-1).sum(-1)

    # Prepare the inputs for the KL divergence calculation
    # (B, n) -> (B, n, 1)
    mu_i = mu_i.unsqueeze(-1)
    mu = mu.unsqueeze(-1)
    mu_delta = mu - mu_i

    # Compute source matrix from the Cholesky factorization
    sigma_i = a_i @ batch_transpose(a_i)  # (B, n, n)
    sigma = a @ batch_transpose(a)  # (B, n, n)

    # Compute the determinants of the covariance matrices
    sigma_i_det = sigma_i.det()  # (B,)
    sigma_det = sigma.det()  # (B,)

    # Clamp the determinants to avoid negative values due to numerical calculation error
    sigma_i_det = torch.clamp_min(sigma_i_det, 1e-6)
    sigma_det = torch.clamp_min(sigma_det, 1e-6)

    # Compute the inverse of the covariance matrices
    sigma_i_inv = sigma_i.inverse()  # (B, n, n)
    sigma_inv = sigma.inverse()  # (B, n, n)

    # Compute the KL divergence
    n = a.size(-1)  # Number of variables
    inner_mu = (batch_transpose(mu_delta) @ sigma_i_inv @ mu_delta).squeeze()  # (B,)
    inner_sigma = torch.log(sigma_det / sigma_i_det) - n + batch_trace(sigma_inv @ sigma_i)  # (B,)

    # Compute the mean KL divergence
    c_mu = 0.5 * torch.mean(inner_mu)
    c_sigma = 0.5 * torch.mean(inner_sigma)
    return c_mu, c_sigma, torch.mean(sigma_i_det), torch.mean(sigma_det)
