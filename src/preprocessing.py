from skimage import io
from sklearn.decomposition import PCA

data_dir = 
coll = io.ImageCollection(data_dir + '/*.png')
img = io.imread("../input/captcha_named/5_1587401160.2.png")
io.imshow(img)

pca = PCA(n_components = 150, whiten=True)