from nano_mlp import *

xs = [
    [2.0, 3.0], 
    [1.0, -1.0], 
    [-3.0, 0.5]
]

ys = [1.0, -1.0, 1.0]

model = MLP(2, [4, 4, 1])

for k in range(200):

    # forward
    ypred = [model(x)[0] for x in xs]

    # loss: squared error
    loss = sum((yp - y)**2 for yp, y in zip(ypred, ys))

    # backward
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # gradient descent
    for p in model.parameters():
        p.data += -0.05 * p.grad

    if k % 20 == 0:
        print(k, loss.data)