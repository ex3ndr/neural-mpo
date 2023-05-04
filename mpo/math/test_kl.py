import unittest
import torch

from kl import categorical_kl, gaussian_kl


class TestCategoricalKL(unittest.TestCase):

    def test_categorical_kl(self):
        # Test case 1
        p1 = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        p2 = torch.tensor([[0.5, 0.5], [0.7, 0.3]])
        expected_output = torch.tensor(0.2670553922653198)

        self.assertAlmostEqual(categorical_kl(p1, p2).item(), expected_output.item(), places=4)
        # self.assertAlmostEqual(categorical_kl(p2, p1).item(), expected_output.item(), places=4)

        # Test case 2
        p1 = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.5, 0.3, 0.1, 0.1]])
        p2 = torch.tensor([[0.5, 0.1, 0.2, 0.2], [0.1, 0.3, 0.3, 0.3]])
        expected_output = torch.tensor(0.3761770725250244)

        self.assertAlmostEqual(categorical_kl(p1, p2).item(), expected_output.item(), places=4)
        # self.assertAlmostEqual(categorical_kl(p2, p1).item(), expected_output.item(), places=4)

    def test_gaussian_kl(self):
        μi = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        μ = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        Ai = torch.tensor([[[1.0, 0.0], [2.0, 1.0]], [[2.0, 0.0], [1.0, 1.0]]])
        A = torch.tensor([[[1.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.5, 1.0]]])

        C_μ, C_Σ, mean_det_Σi, mean_det_Σ = gaussian_kl(μi, μ, Ai, A)
        expected_C_μ = torch.tensor(0.375)
        expected_C_Σ = torch.tensor(0.6534264087677002)
        expected_mean_det_Σi = torch.tensor(2.5)
        expected_mean_det_Σ = torch.tensor(1.0)

        self.assertAlmostEqual(C_μ.item(), expected_C_μ.item(), places=4)
        self.assertAlmostEqual(C_Σ.item(), expected_C_Σ.item(), places=4)
        self.assertAlmostEqual(mean_det_Σi.item(), expected_mean_det_Σi.item(), places=4)
        self.assertAlmostEqual(mean_det_Σ.item(), expected_mean_det_Σ.item(), places=4)


if __name__ == '__main__':
    unittest.main()
