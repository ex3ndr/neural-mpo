import torch
import unittest

from cholesky import cholesky_vector_to_matrix_t, cholesky_vector_size


class TestCholeskyVectorToMatrixT(unittest.TestCase):

    def test_cholesky_vector_to_matrix_t(self):
        # Test case 1
        vector = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        size = 3
        expected_output = torch.tensor([[
            [1.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
            [4.0, 5.0, 6.0]
        ]])

        self.assertTrue(torch.allclose(cholesky_vector_to_matrix_t(vector, size), expected_output))

        # Test case 2
        vector = torch.tensor([[1.0, 2.0, 3.0]])
        size = 2
        expected_output = torch.tensor([[
            [1.0, 0.0],
            [2.0, 3.0]
        ]])

        self.assertTrue(torch.allclose(cholesky_vector_to_matrix_t(vector, size), expected_output))

    def test_cholesky_vector_size(self):
        # Test case 1
        size = 3
        expected_output = 6

        self.assertEqual(cholesky_vector_size(size), expected_output)

        # Test case 2
        size = 2
        expected_output = 3

        self.assertEqual(cholesky_vector_size(size), expected_output)


if __name__ == '__main__':
    unittest.main()
