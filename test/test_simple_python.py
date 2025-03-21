import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import CI

# Test our simple Python backend
class TestSimplePython(unittest.TestCase):
  def setUp(self):
    self.a = Tensor([1, 2, 3, 4], device="SIMPLE_PYTHON")
    self.b = Tensor([2, 3, 4, 5], device="SIMPLE_PYTHON")
    
  def test_add(self):
    result = (self.a + self.b).numpy()
    np.testing.assert_allclose(result, np.array([3, 5, 7, 9]))
    
  def test_mul(self):
    result = (self.a * self.b).numpy()
    np.testing.assert_allclose(result, np.array([2, 6, 12, 20]))
  
  def test_sub(self):
    result = (self.b - self.a).numpy()
    np.testing.assert_allclose(result, np.array([1, 1, 1, 1]))
    
  def test_div(self):
    result = (self.b / self.a).numpy()
    np.testing.assert_allclose(result, np.array([2, 1.5, 4/3, 1.25]))
    
  def test_pow(self):
    # Cast to float since pow requires floating point
    a_float = self.a.float()
    result = (a_float ** 2).numpy()
    np.testing.assert_allclose(result, np.array([1, 4, 9, 16]))
    
  def test_matmul(self):
    # Create separate, smaller tensors with known values for debugging
    a_mat = Tensor([[1.0, 2.0], [3.0, 4.0]], device="SIMPLE_PYTHON")
    b_mat = Tensor([[2.0, 3.0], [4.0, 5.0]], device="SIMPLE_PYTHON")
    
    # Perform matrix multiplication
    c = a_mat @ b_mat
    
    # Get result
    result = c.numpy()
    
    # Print for debugging
    print("A shape:", a_mat.shape)
    print("B shape:", b_mat.shape)
    print("C shape:", c.shape)
    print("Result:", result)
    
    # Calculate expected result
    expected = np.array([[1, 2], [3, 4]]) @ np.array([[2, 3], [4, 5]])
    print("Expected:", expected)
    
    # Verify
    np.testing.assert_allclose(result, expected)
  
  def test_reduce(self):
    result = self.a.sum().numpy()
    np.testing.assert_allclose(result, 10)
    
  def test_compare(self):
    result = (self.a < self.b).numpy()
    np.testing.assert_allclose(result, np.array([True, True, True, True]))
    
  def test_where(self):
    cond = Tensor([True, False, True, False], device="SIMPLE_PYTHON")
    result = cond.where(self.a, self.b).numpy()
    np.testing.assert_allclose(result, np.array([1, 3, 3, 5]))
    
  @unittest.skipIf(CI, "Minimize test load in CI")
  def test_larger_tensor(self):
    a = Tensor.randn(64, 64, device="SIMPLE_PYTHON")
    b = Tensor.randn(64, 64, device="SIMPLE_PYTHON")
    c = a @ b
    result = c.numpy()
    expected = a.numpy() @ b.numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    
if __name__ == "__main__":
  unittest.main()