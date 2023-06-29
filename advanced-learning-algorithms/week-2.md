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

1. specify how to compute output given input $x$ and parameters $w, b$ (define model)
