from torchvision.datasets import LSUN
from torchvision import transforms
import numpy as np
from keras.preprocessing import image
from tqdm import tqdm

def main(W):
    classes = ['bedroom', 'kitchen', 'conference_room', 'dining_room', 'church_outdoor']
    N = 100000
    image_size = (W, W)
    rootdir = '.'

    for i, c in enumerate(classes):
        print(c)
        lsun = LSUN(rootdir, ['%s_train' % c], transform=transforms.Compose([transforms.Resize(image_size)]))
        for n in tqdm(range(N)):
            lsun[n][0].save('data_split_%d/%s/0/%07d.jpg' % (W, c, n))
            lsun[n][0].save('data_%d/0/%07d.jpg' % (W, n + i * N))


if __name__ == '__main__':
    main(W=32)
    main(W=64)
