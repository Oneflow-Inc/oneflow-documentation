import oneflow as flow
import numpy as np 
from PIL import Image
from quickstart import Net
import sys

img=Image.open(sys.argv[1])
np_img = np.array(img)

loaded_model = Net(784, 128, 64, 10)
loaded_model.load_state_dict(flow.load("mnist_model"))
flow_img = flow.Tensor(np_img)
pred = loaded_model(flow_img.reshape((-1,28*28)))

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
x = pred[0].argmax().numpy()
x_real = classes[x.item(0)]

print(x_real)

