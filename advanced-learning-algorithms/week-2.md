# Week 2 - Neural network training

## Tensorflow implementation

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

model = Sequential([
                    Dense(units=25, activation='sigmoid'),
                    Dense(units=15, activation='sigmoid')
                    Dense(units=1, activation='sigmoid')
])

model.compile(loss=BinaryCrossentropy())
model.fit(X, Y, epochs=100)
```

_tip)_ `epochs` is the number of steps in gradient decent

## Training Details

1. Specify how to compute output given input $x$ and parameters $w, b$ (define model)

$f_{\vec{w}, b}(\vec{x}) = ?$

```python
model = Sequential([
                    Dense(units=25, activation='sigmoid'),
                    Dense(units=15, activation='sigmoid')
                    Dense(units=1, activation='sigmoid')
])
```

1. Specify loss and cost function

Loss function: $L(f_{\vec{w}, b}(\vec{x}), y)$

Cost function: $J(\vec{w}, b) = \dfrac{1}{m}\Sigma_{i = 1}^{m}{L(f_{\vec{w}, b}(\vec{x}), y)}$

```python
model.compile(loss=BinaryCrossentropy())
```

3. Train on data to minimize $J(\vec{w}, b)$

This will run the gradient decent algorithm for every unit of every layer to get the minimal parameters($W$ and $b$)

```python
model.fit(X, Y, epochs=100)
```

## Alternatives to the sigmoid activation

$$
\begin{cases}
    \text{1. Linear} \quad g(z) = z \\
    \text{2. Sigmoid} \quad g(z) = \dfrac{1}{1 + e^{-z}} \\
    \text{3. ReLU} \quad g(z) = max(0, z)
\end{cases}
$$

## Choosing an activation function

Most of the time there is a fairly natural choice for the last layers neuron based on the output type. for example if it's a binary classification, it would be Sigmoid, and if it's Regression it would be either ReLU or Linear.

The ReLU activation has replaces the Sigmoid as the most popular activation function for the following reasons:

1. ReLU is faster to compute as it's simpler than Sigmoid
2. Sigmoid goes flat in two places(start and end) whereas ReLU goes flat in just one place($z \leq 0$); so the Gradient Decent performs a lot slower in ReLU
3. ReLU learns faster as the hidden layer

## Why do we need activation function

Here is the reason why a neural network with just Linear activation functions wont work:

Suppose we have a two layer neural network with 1 Linear neuron per each layer:
$$
\begin{cases}
    a_1^{[1]} = \vec{w_1}^{[1]} \cdot \vec{x} + \vec{b_1}^{[1]} \\
    a_1^{[2]} = \vec{w_1}^{[2]} \cdot \vec{a}^{[1]} + \vec{b_1}^{[2]}
\end{cases}
\quad \Longrightarrow \quad
a_1^{[2]} = \vec{w_1}^{[2]}(\vec{w_1}^{[1]}x + \vec{b_1}^{[1]}) + \vec{b_1}^{[2]} \quad = \quad
(\vec{w_1}^{[2]} \vec{w_1}^{[1]})x + \vec{w_1}^{[2]} \vec{b_1}^{[1]} + \vec{b_1}^{[2]} \quad = \quad \bold{wx + b}
$$
So the result would be just another Linear function.

Similarly using Linear activation function as all hidden layers and using a Sigmoid activation function as output layer would result in a simple Sigmoid function.

_tip)_ Don't just use just the Linear activation functions in hidden layers.  
_tip)_ Using just ReLU as hidden layers would do just fine

## Softmax

Given an input, Softmax calculates the probability of that input being under each of the existing categories.

$N$ = number of categories
$$
\begin{cases}
    z_i = \vec{w_i} \cdot \vec{x} + b_i \quad i = 1, 2, 3, ...,N\\
    a_i = \dfrac{e^{z_i}}{\Sigma_{j=1}^N\ e^{z_j}} = P(y = i|\vec{x})
\end{cases}
$$

The $\Sigma_{j=1}^N\ a_j$ will be equal to $1$.

_tip)_ If we apply the Softmax regression for $N = 2$, it ends up computing the same thing as Logistic regression(although the parameters will be a little bit different)

### Cost function of Softmax

$$
loss(a_1, ..., a_N, y) =
\begin{cases}
    -\log{a_1} \qquad \text{if $y = 1$} \\
    -\log{a_2} \qquad \text{if $y = 2$} \\
    \dot{\dot{.}}
\end{cases}
$$

## Neural network with Softmax output

Since the softmax only computes the probability of one given category, we will need as many neurons as existing categories in our last Softmax layer e.g. for handwritten number classification we will need 10 Softmax neurons.

_note)_ In other activation functions $a_i$ was the function of $z_i$, but in Softmax, each $a$ is a function of all available $z$

### MNIST handwritten digit recognition with Tensorflow

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model = Sequential([
                    Dense(units=25, activation='relu'),
                    Dense(units=15, activation='relu')
                    Dense(units=10, activation='softmax')
])

# sparse means that y can only take one of the categories
model.compile(loss=SparseCategoricalCrossentropy())
model.fit(X, Y, epochs=100)
```

> Even though this code works, there is a better implementation for this purpose.

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

model = Sequential([
                    Dense(units=25, activation='sigmoid'),
                    Dense(units=15, activation='sigmoid')
                    Dense(units=10, activation='linear')
])

model.compile(loss=BinaryCrossentropy(from_logits=True))
model.fit(X, Y, epochs=100)

logit = model(X)

f_x = tf.nn.sigmoid(logit)
```

## Classification with multiple outputs

For example in an application that needs to predict whether there are cars and busses and people in an image; So the output should be a horizontal $3 \times 1$ matrix that has a binary number in each row.

So the final layer has three Sigmoid neurons.

_note)_ This is called _Multi label classification_
