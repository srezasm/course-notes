# Getting Started with pandas

## Series

A Series is a one-dimensional array-like object containing a sequence of values.

```python
pd.Series([4, 7, -5, 3])
```

The string representation of a Series displayed interactively shows the index on the left and the values on the right.

Series with an index identifying each data point with a label:

```python
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
```

You can use labels in the index when selecting single values or a set of values:

```python
obj2['a'] = 6
obj2[['c', 'a', 'd']]
```

Using NumPy functions or NumPy-like operations, such as filtering with a boolean array, scalar multiplication, or applying math functions, will preserve the index-value link:

```python
obj2[obj2 > 0] # print items that are grater than 0

obj2 * 2 # multiply all items by 2

np.exp(obj2) # bring each item to the power of Euler's number
```

Series is as a fixed-length, ordered dict, as it is a mapping of index values to data values.

```python
'a' in obj2 # True
'x' in obj2 # False
```

You can even convert a dict into Series directly:

```python
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)
```

Change the order of indexes:

```python
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
```

Here, three values found in sdata were placed in the appropriate locations, but since no value for 'California' was found, it appears as NaN (not a number), which is considered in pandas to mark missing or NA values.

The `isnull` and `notnull` functions in pandas should be used to detect missing data:

```python
pd.isnull(obj4) # smae as obj4.isnull()

# California    True
# Ohio          False
# Oregon        False
# Texas         False
# dtype: bool

pd.notnull(obj4) # smae as obj4.notnull()

# California    False
# Ohio          True
# Oregon        True
# Texas         True
# dtype: bool
```

Both the Series object itself and its index have a name attribute, which integrates with other key areas of pandas functionality:

```python
obj4.name = 'population'
obj4.index.name = 'state'

# state
# California    NaN
# Ohio          35000.0
# Oregon        16000.0
# Texas         71000.0
# Name: population, dtype: float64
```
