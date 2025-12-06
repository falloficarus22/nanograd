from bitgrad import MLP, Value
from bitgrad.viz import draw_graph

model = MLP(2, [4, 4, 1])
x = [2.0, 3.0]
y = 1.0

yp = model(x)[0]
loss = (yp - y) ** 2
loss.backward()

dot = draw_graph(loss)
dot.render('graph', view = True)