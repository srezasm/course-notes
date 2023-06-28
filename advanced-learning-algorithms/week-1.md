# Week 1 - Neural Networks

## Neural network layer

A simple visualization of a deep learning neural network with an input ($x$) and a hidden layer and an output($\vec{a}^{[2]}$) with Sigmoid function as activation of all layers.

$$
x \overset{\vec{x}}{\Longrightarrow}

\begin{cases}
\vec{w_1}^{[1]}, \vec{b_1}^{[1]} \quad a_1^{[1]} = g(\vec{w_1}^{[1]} \cdot \vec{x} + \vec{b_1}^{[1]}) \\
\vec{w_2}^{[1]}, \vec{b_2}^{[1]} \quad a_2^{[1]} = g(\vec{w_2}^{[1]} \cdot \vec{x} + \vec{b_2}^{[1]}) \\
\vec{w_3}^{[1]}, \vec{b_3}^{[1]} \quad a_3^{[1]} = g(\vec{w_3}^{[1]} \cdot \vec{x} + \vec{b_3}^{[1]})
\end{cases}

\enspace \overset{\vec{a}^{[1]}}{\Longrightarrow}

\begin{cases}
\vec{w_1}^{[2]}, \vec{b_1}^{[2]} \quad a_1^{[2]} = g(\vec{w_1}^{[2]} \cdot \vec{a}^{[1]} + \vec{b_1}^{[2]})

\enspace \overset{\vec{a}^{[2]}}{\Longrightarrow} [1, 0]
\end{cases}
$$

## More complex neural network

General form: $a_j^{[l]} = g(\vec{w_j}^{[l]} \cdot \vec{a^{[l - 1]}} + \vec{b_j}^{[l]})$
Where $l$ is an arbitrary layer and $j$ is an arbitrary unit.

## Data in tensorflow

Convert Tensor object to Numpy array by

```python
a1 = [[1, 2], [2, 4]] # some data
layer2 = Dense(units=1, activation='sigmoid')
a2 = layer2(a1)

np_arr = a2.numpy() # returns the numpy array
```

## Building a neural network


