# Week 2 - Recommender systems

## Making recommendations

| Movie                | Alice(1) | Bob(2) | Carol(3) | Dave(4) |
|----------------------|----------|--------|----------|---------|
| Love at last         | 5        | 5      | 0        | 0       |
| Romance forever      | 5        | ?      | ?        | 0       |
| Cute puppies of love | ?        | 4      | 0        | ?       |
| Nonstop car chases   | 0        | 0      | 5        | 4       |
| Swords vs. karate    | 0        | 0      | 5        | ?       |

Given the dataset above, we define the following notations:

- $n_u$ = no. of users
- $n_m$ = no. of movies
- $r(i, j) = 1$ if user $j$ has rated movie $i$
- $y^{(i, j)}$ = rating given by user $j$ to movie $i$ (defined only if $r(i, j) = 1$)

Now, given the users rated movies, we can predict the rating of other movies for that user and recommend him/her the highest predicted rating.

## Using per-item features

Lets add two features for movies

| Movie                | Alice(1) | Bob(2) | Carol(3) | Dave(4) | $x_1$ <br> (romance) | $x_2$ <br> (action) |
|----------------------|----------|--------|----------|---------|----------------------|---------------------|
| Love at last         | 5        | 5      | 0        | 0       | 0.9                  | 0                   |
| Romance forever      | 5        | ?      | ?        | 0       | 1.0                  | 0.01                |
| Cute puppies of love | ?        | 4      | 0        | ?       | 0.99                 | 0                   |
| Nonstop car chases   | 0        | 0      | 5        | 4       | 0.1                  | 1.0                 |
| Swords vs. karate    | 0        | 0      | 5        | ?       | 0                    | 0.9                 |

Notation:

- $r(i, j) = 1$ if user $j$ has rated movie $i$ ($0$ otherwise)
- $y^{(i, j)}$ = rating given by user $j$ to movie $i$ (if defined)
- $W^{(i)}, b^{(i)}$ = parameters for user $j$
- $x^{(i)}$ = feature vector for movie $i$
- $m^{(j)}$ = no. of movies rated by user $j$

For user $j$ and movie $i$, predict rating: $W^{(j)} \cdot x^{(i)} + b^{(j)}$

**Cost function:**

$$
\large \underset{W^{(j)} b^{(j)}}{\text{min}} J(W^{(j)}, b^{(j)}) = \dfrac{1}{2} \displaystyle \sum_{i:r(i,j) = 1} \left(W^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)}\right)^2 + \dfrac{\lambda}{2} \displaystyle \sum_{k=1}^{n} \left(W_{k}^{(j)}\right)^2
$$

_note)_ The $\displaystyle \sum_{i:r(i,j) = 1}$ denotes the sum of all movies $i$ that user $j$ has already rated

In recommendation systems, since $m^{(j)}$ is just a constant, it's convenient to use $\dfrac{\lambda}{2}$ instead of $\dfrac{\lambda}{2m^{(j)}}$ in divisions.

Now to learn parameters $W^{(1)}, b^{(1)}, \dotsb, W^{(n_u)}, b^{(n_u)}$ for all users:

$$
\large J
\begin{pmatrix}
    W^{(1)}, \dotsb, W^{(n_u)} \\
    b^{(1)}, \dotsb, b^{(n_u)} \\
\end{pmatrix}
= \dfrac{1}{2} \displaystyle \sum_{j=1}^{n_u} \displaystyle \sum_{i:r(i,j) = 1} \left(W^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)}\right)^2 + \dfrac{\lambda}{2} \displaystyle \sum_{j=1}^{n_u} \displaystyle \sum_{k=1}^{n} \left(W_{k}^{(j)}\right)^2
$$

## Collaborative filtering algorithm

### Description

What if we didn't have the $x_1$ and $x_2$ features?

| Movie                | Alice(1) | Bob(2) | Carol(3) | Dave(4) | $x_1$ <br> (romance) | $x_2$ <br> (action) |
|----------------------|----------|--------|----------|---------|----------------------|---------------------|
| Love at last         | 5        | 5      | 0        | 0       | ?                    | ?                   |
| Romance forever      | 5        | ?      | ?        | 0       | ?                    | ?                   |
| Cute puppies of love | ?        | 4      | 0        | ?       | ?                    | ?                   |
| Nonstop car chases   | 0        | 0      | 5        | 4       | ?                    | ?                   |
| Swords vs. karate    | 0        | 0      | 5        | ?       | ?                    | ?                   |

Where we don't have any feature for examples, but instead we have the users interaction for examples, we can use users interactions as a feature to even predict the feature users interactions and this is _collaborative filtering_.

In this example, although we don't have any features corresponding to each movie, we have the users ratings, and using these ratings beside the target users ratings, we can predict the feature ratings of the target user!

Given $W^{(1)}, b^{(1)}, \dotsb, W^{(n_u)}, b^{(n_u)}$  
to learn $x^{(i)}$:

$$
\large J(x^{(i)}) = \dfrac{1}{2} \displaystyle \sum_{j:r(i,j) = 1} \left(W^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)}\right)^2 + \dfrac{\lambda}{2} \displaystyle \sum_{k=1}^{n} \left(x_{k}^{(i)}\right)^2
$$

_note)_ Notice unlike the $J(W^{(j)}, b^{(j)})$, we used $\displaystyle \sum_{j:r(i,j) = 1}$, to determine the sum all users $j$ that have rated the movie $i$ and the regularization term is also different

Now to learn $x^{(1)}, \dotsb, x^{(n_m)}$:

$$
\large J(x^{(1)}, \dotsb, x^{(n_m)}) = \dfrac{1}{2} \displaystyle \sum_{i=1}^{n_m} \displaystyle \sum_{j:r(i,j) = 1} \left(W^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)}\right)^2 + \dfrac{\lambda}{2} \displaystyle \sum_{i=1}^{n_m} \displaystyle \sum_{k=1}^{n} \left(x_{k}^{(i)}\right)^2
$$

### Algorithm

Cost function to learn $W^{(1)}, b^{(1)}, \dotsb, W^{(n_u)}, b^{(n_u)}$:

$$
\large \underset{W^{(1)}, b^{(1)}, \dotsb, W^{(n_u)}, b^{(n_u)}}{\text{min}}\ \dfrac{1}{2} \displaystyle \sum_{j=1}^{n_u} \displaystyle \sum_{i:r(i,j) = 1} \left(W^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)}\right)^2 + \dfrac{\lambda}{2} \displaystyle \sum_{j=1}^{n_u} \displaystyle \sum_{k=1}^{n} \left(W_{k}^{(j)}\right)^2
$$

Cost function to learn $x^{(1)}, \dotsb, x^{(n_m)}$:

$$
\large \underset{x^{(1)}, \dotsb, x^{(n_m)}}{\text{min}}\ \dfrac{1}{2} \displaystyle \sum_{i=1}^{n_m} \displaystyle \sum_{j:r(i,j) = 1} \left(W^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)}\right)^2 + \dfrac{\lambda}{2} \displaystyle \sum_{i=1}^{n_m} \displaystyle \sum_{k=1}^{n} \left(x_{k}^{(i)}\right)^2
$$

Put them together:

$$
\large \underset{\begin{pmatrix} W^{(1)}, \dotsb, W^{(n_u)} \\ b^{(1)}, \dotsb, b^{(n_u)} \\ x^{(1)}, \dotsb, x^{(n_m)} \end{pmatrix}}{\text{min}}\ \dfrac{1}{2} \displaystyle \sum_{(i,j):r(i,j) = 1} \left(W^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)}\right)^2 + \dfrac{\lambda}{2} \displaystyle \sum_{j=1}^{n_u} \displaystyle \sum_{k=1}^{n} \left(W_{k}^{(j)}\right)^2 + \dfrac{\lambda}{2} \displaystyle \sum_{i=1}^{n_m} \displaystyle \sum_{k=1}^{n} \left(x_{k}^{(i)}\right)^2
$$

### Gradient Descent

In collaborative filtering, the cost function is a function of $W$, $b$ and $x$; so the optimization function should be so:

repeat: {
    
- $W_{i}^{(j)} = W_{i}^{(j)} - \alpha \dfrac{\partial}{\partial W_{i}^{(j)}} J\left(W, b, x\right)$
- $b^{(j)} = b^{(j)} - \alpha \dfrac{\partial}{\partial b^{(j)}} J\left(W, b, x\right)$
- $x_{k}^{(i)} = x_{k}^{(i)} - \alpha \dfrac{\partial}{\partial x_{k}^{(i)}} J\left(W, b, x\right)$

}

## Binary labels: favs, likes and clicks

In real-world applications, there are a lot of other features besides rating that can be used in recommendation systems and collaborative filtering.

| Movie                | Alice(1) | Bob(2) | Carol(3) | Dave(4) |
|----------------------|----------|--------|----------|---------|
| Love at last         | 1        | 1      | 0        | 0       |
| Romance forever      | 1        | ?      | ?        | 0       |
| Cute puppies of love | ?        | 1      | 0        | ?       |
| Nonstop car chases   | 0        | 0      | 1        | 1       |
| Swords vs. karate    | 0        | 0      | 1        | ?       |

Here are some examples of what the binary numbers can mean:

- Did user $j$ purchase an item after being shown?
- Did user $j$ fav/like an item?
- Did user $j$ spend at least 30sec with an item?
- Did user $j$ click on an item?

### From regression to binary classification

- Previously:
  - Predict $y^{(i,j)}$ as $W^{(j)} \cdot x^{(i)} + b^{(j)}$
- For binary classification:
  - Predict that the probability of $y^{(i,j)} = 1$ is given by $g(W^{(j)} \cdot x^{(i)} + b^{(j)})$
  - Where $g(z) = \dfrac{1}{1 + e^{-z}}$

### Cost function for binary classification

- Previous cost function:
  - $\dfrac{1}{2} \displaystyle \sum_{(i,j):r(i,j) = 1} \left(W^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)}\right)^2 + \dfrac{\lambda}{2} \displaystyle \sum_{j=1}^{n_u} \displaystyle \sum_{k=1}^{n} \left(W_{k}^{(j)}\right)^2 + \dfrac{\lambda}{2} \displaystyle \sum_{i=1}^{n_m} \displaystyle \sum_{k=1}^{n} \left(x_{k}^{(i)}\right)^2$
- Loss for binary labels: $y^{(i,j)}: f_{{W,b,x}}(x) = g(W^{(j)} \cdot x^{(i)} + b^{(j)})$
  - $L(f_{{W,b,x}}(x), y^{(i,j)}) = - y^{(i,j)} \log \left(f_{{W,b,x}}(x)\right) - \left(1 - y^{(i,j)}\right) \log \left(1 - f_{{W,b,x}}(x) \right)$
  - $J(W, b, x) = \sum_{(i,j):r(i,j) = 1} \left(f_{{W,b,x}}(x), y^{(i,j)} \right)$
  - Where $f_{{W,b,x}}(x) = g\left(W^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)}\right)$

## Mean normalization

First we add a new user Eve with no ratings:

| Movie                | Alice(1) | Bob(2) | Carol(3) | Dave(4) | Eve(2) |
|----------------------|----------|--------|----------|---------|--------|
| Love at last         | 5        | 5      | 0        | 0       | ?      |
| Romance forever      | 5        | ?      | ?        | 0       | ?      |
| Cute puppies of love | ?        | 4      | 0        | ?       | ?      |
| Nonstop car chases   | 0        | 0      | 5        | 4       | ?      |
| Swords vs. karate    | 0        | 0      | 5        | ?       | ?      |

The parameters that the current system will come up are: $W^{(5)} = [0, 0] \quad b^{(5)} = 0$. So the Eves predicted ratings for all movies will be $0$, and it's not good.

### Algorithm

We will take all of ratings into a matrix and take the average of each row into a new vector $\mu$, and finally subtract the $\mu$ from initial ratings matrix:

$$
\begin{bmatrix}
    5 & 5 & 0 & 0  & ? \\
    5 & ? & ? & 0  & ? \\
    ? & 4 & 0 & ?  & ? \\
    0 & 0 & 5 & 4  & ? \\
    0 & 0 & 5 & ?  & ? \\
\end{bmatrix}
\qquad \mu =
\begin{bmatrix}
    2.5 \\
    2.5 \\
    2 \\
    2.25 \\
    1.25 \\
\end{bmatrix}
\qquad
\begin{bmatrix}
2.5   & 2.5   & -2.5 & -2.5  & ? \\
2.5   & ?     & ?    & -2.5  & ? \\
?     & 2     & -2   & ?     & ? \\
-2.25 & -2.25 & 2.75 & 1.75  & ? \\
-2.25 & -1.25 & 3.75 & -1.25 & ? \\
\end{bmatrix}
$$

And now the final matrix will be our new $y^{(i,j)}$, and the system will learn based on it.

Furthermore, because we subtracted $\mu$, for user $j$ on movie $i$, we predict: $W^{(j)} \cdot x^{(i)} + b^{(j)} {\bf + \mu_{i}}$

Now even though we still have initial parameters of $W^{(5)} = [0, 0] \quad b^{(5)} = 0$ for Eve, the resulting prediction will be $0 - \mu$, on which we can show him the highest rated movies at first.

_note)_ We can take the averages from columns too, but that would be useful when we want to predict the features of a new movie instead of a new user

## Tensorflow implementation of collaborative filtering

### Custom training loop

```python
w = tf.Variable (3.0)
x = 1.0
y = 1.0 # target value
alpha = 0.01

iterations = 30
for iter in range(iterations):
    # Use TensorFlow's Gradient tape to record the steps
    # used to compute the cost J, to enable auto differentiation.
    with tf.Gradient Tape () as tape:
        fwb = w * x
        costJ (fwb - y) **2

    # Use the gradient tape to calculate the gradients
    # of the cost with respect to the parameter w.
    [dJdw] = tape.gradient(costJ, [w])

    # Run one step of gradient descent by updating
    # the value of w to reduce the cost.
    w.assign_add(-alpha * dJdw)
```

### Implementation in TensorFlow

```python
#Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-1)
iterations = 200
for iter in range (iterations):
    # Use TensorFlow's GradientTape
    # to record the operations used to compute the cost
    with tf.GradientTape () as tape:
    
        # Compute the cost (forward pass is included in cost)
        cost_value = cofiCostFuncV(X, W, b, Ynorm, R, num_users, num_movies, lambda)
    
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss
    grads = tape.gradient(cost_value, [X, W, b])

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, [X, W, b]))
```

## Finding related items

The features $x^{(i)}$ of item $i$(user, movie, etc.) are quite hard to interpret. For example in the case of movies, it's hard to look at each movie individually and detect how much romance, action or any other genera it is. But collectively, we can predict its features.

To find other items related to it, find item $k$ with $x^{(k)}$ similar to $x^{(i)}$ with calculating their _squared distances_.

$$\displaystyle \sum_{l=1}^{n} \left(x_{l}^{(k)} - x_{l}^{(i)}\right)^2$$

The squared distance in mathematics is also shown as: $||x^{(k)} - x^{(i)}||^2$

### Limitations of collaborative filtering

Cold start problem. How to

- rank new items that few users have rated?
- show something reasonable to new users who have rated few items?

Use side information about items or users:

- Item: Genre, movie stars, studio, .
- User: Demographics (age, gender, location), expressed preferences, ...
