# Advise for applying machine learning

## Evaluating a model

We could detect the models issues by plotting the features and the predicting line, but as the number of features grows, plotting becomes harder.

Split the dataset into _training set_ and _test set_. Train the model on _training set_ and then evaluate the models performance by test set.

1. Fit parameters by minimizing cost function $J(\vec{w}, b)$

    $$J(\vec{w}, b) = \left[\dfrac{1}{2m_{train}}\displaystyle\sum_{i=1}^{m_{train}}{(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2}+\dfrac{\lambda}{2m_{train}}\displaystyle\sum_{j=1}^{n}{w_j^2}\right]$$

2. Compute test error

    $$J_{test}(\vec{w}, b) = \left[\dfrac{1}{2m_{test}}\displaystyle\sum_{i=1}^{m_{test}}{(f_{\vec{w},b}(\vec{x}_{test}^{(i)})-y_{test}^{(i)})^2}\right]$$

3. Compute training error

    $$J_{train}(\vec{w}, b) = \left[\dfrac{1}{2m_{train}}\displaystyle\sum_{i=1}^{m_{train}}{(f_{\vec{w},b}(\vec{x}_{train}^{(i)})-y_{train}^{(i)})^2}\right]$$

If the $J_{train}$ is optimal and $J_{test}$ is high, the model's probably overfitting.

The same 1, 2 and 3 steps goes for classification models too, except the $J()$ function will change accordingly.  
there's also a more common way to calculate the $J_{train}$ and $J_{test}$ for classification problems:

$J_{train}$ is the fraction of _training set_ that has been misclassified.  
$J_{test}$ is the fraction of _test set_ that has been misclassified.

## Model selection and Training/Cross validation/Test sets

To choose the best model architecture amongst the models that we have defined, we could calculate the $J_{test}$ of all models and select the one with the least loss, but what if that architecture overfits and only performs well on the test set that you have provided?

In this approach, we split the model into 3 sets: _Training set_, _Cross validation set_, _Test set_. Then we train the models on _training set_ and select the best model based on _Cross validation error_ and then we could evaluate the accuracy of selected model by calculating the _Test set error_.

_tip)_ Cross validation set is also called _validation set_, _development set_ and _dev set_.  

_tip)_ We show the Cross validation cost function as $J_{cv}$.

_tip)_ Assign most of the data to the _training set_ and divide the rest between the _CV_ and _test_ sets

Training error: $\qquad J_{train}(\vec{w}, b) = \dfrac{1}{2m_{train}}\left[\displaystyle\sum_{i=1}^{m_{train}}{(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2}\right]$

Cross validation error: $\qquad J_{cv}(\vec{w}, b) = \dfrac{1}{2m_{cv}}\left[\displaystyle\sum_{i=1}^{m_{cv}}{(f_{\vec{w},b}(\vec{x}_{cv}^{(i)})-y_{cv}^{(i)})^2}\right]$

Test error: $\qquad J_{test}(\vec{w}, b) = \dfrac{1}{2m_{test}}\left[\displaystyle\sum_{i=1}^{m_{test}}{(f_{\vec{w},b}(\vec{x}_{test}^{(i)})-y_{test}^{(i)})^2}\right]$

It worth nothing that if the selected model performs bad on the _test set_ we can conclude that it's overfits and we can select the nex-best-model in _cross validation error_.

## Diagnosing bias and variance

Looking at the _bias_ and _variance_ of a model gives a really good insight of how the model is performing, but we can't easily plot the output of a model in order to detect high or low bias and variance; therefore a more systematic way to diagnose the algorithm would be to look at the performance of algorithm on the _training_ and _CV_ sets.

![image1](assets/img-1.jpg)

<img src="./assets/img-2.png" align="right" height="250px">

High bias:  
$J_{train}$ is high  
$J_{train} \approx J_{cv}$

High variance:  
$J_{cv} \gg J_{train}$  
$J_{train}$ may be low

High bias and variance(rare situation, where a part of the input _overfits_ and another part _underfits_):  
$J_{train}$ is high  
$J_{cv} \gg J_{train}$  
