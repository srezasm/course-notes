# NumPy Basics: Arrays and Vectorized Computation

## File Input and Output with Arrays

Save numpy array to binary file:

```python
arr = np.arange(10)
np.save('file_name', arr)
```

`.npy` will be automatically appended to the file name

Load array from file:

```python
np.load('file_name.npy')
```

Save multiple arrays to an uncompressed file:

```python
np.savez('file_name.npz', a1=np.arange(10), a2=np.arange(10, 20))
```

Load as a dict object:

```python
arch = np.load('file_name.npz')

a1 = arch['a1']
a2 = arch['a2']
```

If the data compresses well, use the bellow function:

```python
np.savez_compressed('file_name.npz', a1=np.arange(10), a2=np.arange(10, 20))
```

The loading is the same as uncompressed file.

## Linear Algebra

`numpy.linalg` has a standard set of decompositions and things like inverse and determinant. These are implemented under the hood via the same industry-standard linear algebra libraries used in other languages like MATLAB and R, such as BLAS, LAPACK, or possibly (depending on your NumPy build) the proprietary Intel MKL (Math Kernel Library):

```python
from numpy.linalg import inv, qr
```

## Pseudorandom Number Generation

`numpy.random` uses the built-in python `random` to generate random numbers a lot more efficiently.

```python
samples = np.random.normal(size=(4, 4))
```

generates a 4\*4  2D normal matrix.

these are _pseudorandom_ numbers though, because they use a deterministic algorithm that generates numbers based on the seed of the random number generator. You can change the random number generators seed as following:

```python
np.random.seed(1234)
```

The data generation function in `numpy.random` use a global random seed. To avoid global state, use `numpy.random.RandomState` to create a random number generator isolated from the others:

```python
rng = np.random.RandomState(1234)
rng.randn(10) # returns 10 random numbers in an array
```

Example implementation of a random walk application in pure python and numpy way:

```python
# Pure python
import random
position = 0
for i in range(1000):
	step = 1 if random.randint(0, 1) else -1
	position += step

# Numpy
draws = np.random.randint(0, 2, size=1000) # generate array of random 0 & 1 with the length of steps
steps = np.where(draws == 1, 1, -1) # create steps array and fill each item based on the value of corresponding index at draws array; 1 if draws == 1 else -1
walk = steps.cumsum() # fill the walk array with the sum of each item in steps with the previous item
```
