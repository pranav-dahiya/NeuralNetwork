from tqdm import tqdm
import numpy as np
import gzip


def extract_data(filename, num_images, IMAGE_WIDTH=28):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        x = [image/255 for image in data]
        return x


def extract_labels(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        y = np.array([[int(i==x) for i in range(10)] for x in labels])
    return y


def sigmoid(x):
    return np.divide(np.exp(x),(np.exp(x)+1))


def sum_of_squares(label, output):
    return np.sum((label-output)**2)


images = extract_data("t10k-images-idx3-ubyte.gz", 10000)
labels = extract_labels("t10k-labels-idx1-ubyte.gz", 10000)

W = np.array([np.random.rand(images[0].size,500), np.random.rand(500,10)])
B = np.array([np.random.rand(500), np.random.rand(10)])

dW = np.array([np.zeros((images[0].size,500)), np.zeros((500,10))])
dB = np.array([np.zeros((500)), np.zeros((10))])

progress_bar = tqdm(range(1000))

for _ in progress_bar:
    for label,image in zip(labels,images):
        #feedforward
        hidden_layer = sigmoid(image.dot(W[0])+B[0])
        output = sigmoid(hidden_layer.dot(W[1])+B[1])
        #show output
        progress_bar.set_description("Cost: %.2f" % sum_of_squares(label,output))
        #backpropagation
        dW[1] = np.outer(hidden_layer,2*(label-output)*output*(1-output))
        dX = W[1].dot(2*(label-output)*output*(1-output))
        dB[1] = (2*(label-output)*output*(1-output))
        dW[0] = np.outer(image,dX*hidden_layer*(1-hidden_layer))
        dB[0] = dX*hidden_layer*(1-hidden_layer)
        #updation
        W += dW
        B += dB

np.savetxt("weights",W,delimiter=',')
np.savetxt("biases",B,delimiter=',')
