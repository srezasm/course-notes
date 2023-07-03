# Decision trees

## Decision tree model

- Each cell in the decision tree is called a _node_.
- The top most node is called the _root node_.
- The all other nodes, excluding the bottom most nodes are called _decision nodes_.
- The bottom most nodes are called _leaf nodes_.

<image src="./assets/img-10.jpg" height="300px">

Based on the count of features and possible value of each feature, there are several decision trees for each application. The job of a decision tree algorithm is to pick the one that does the best on the training set and also generalizes well to new data.

## Learning process

First off, we need to pick the root node feature.  
Then we should look at the left node and decide what feature to use next and so on till we pick the left most branch features.  
As for the left, we continue the process for the right branch and select the features for them as well.

1. How to choose what feature to split at each node? Select by the maximum purity(purity of a feature is the precision of that feature)
2. When do you stop splitting?
    - When a node is 100% one class.
    - when splitting a node will result in the tree exceeding a maximum depth(depth is the number of hops that takes to reach the selected node from root node)
        - Keeping the maximum depth small prevents the tree to become too big.
        - Keeping maximum depth small makes it less prone to overfitting.
    - When improvements in purity score are below a threshold.
    - When the number of examples in a node is below a threshold.
