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
    
- $W^{(j)}_{i} = W^{(j)}_{i} - \alpha \dfrac{\partial}{\partial W^{(j)}_{i}} J\left(W, b, x\right)$
- $b^{(j)} = b^{(j)} - \alpha \dfrac{\partial}{\partial b^{(j)}} J\left(W, b, x\right)$
- $x^{(i)}_{k} = x^{(i)}_{k} - \alpha \dfrac{\partial}{\partial x^{(i)}_{k}} J\left(W, b, x\right)$

}
