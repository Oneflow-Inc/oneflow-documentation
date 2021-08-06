import oneflow as flow

x = flow.tensor(2., requires_grad=True)
y = flow.tensor(3., requires_grad=True)
z = x*y 

x_grad = flow.autograd.grad(z,x,retain_graph=True)
y_grad = flow.autograd.grad(z,y)

print(x_grad[0],y_grad[0])


