import numpy as np

ITER_COUNT = 500
LR = 0.01

def forward(x, w):
    return np.matmul(x, w)

def loss(y_pred, y):
    return (0.5*(y_pred-y)**2).sum()

def gradient(x, y, y_pred):
    return np.matmul(x.T, (y_pred-y))

if __name__ == "__main__":
    # train data: Y = 2*X1 + 3*X2
    x = np.array([[1, 2], [2, 3], [4, 6], [3, 1]], dtype=np.float32)
    y = np.array([[8], [13], [26], [9]], dtype=np.float32)

    w = np.random.rand(2, 1)
    for i in range(0, ITER_COUNT):
        y_pred = forward(x, w)
        l = loss(y_pred, y)
        if (i+1) % 50 == 0: print(f"{i+1}/{500} loss:{l}")

        grad = gradient(x, y, y_pred)
        w -= LR*grad

    print(f"w:{w}")