#Noa Azoulay
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.spatial.distance import cdist




def init_centroids(X, K):
    """
    Initializes K centroids that are to be used in K-Means on the dataset X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    K : int
        The number of centroids.

    Returns
    -------
    centroids : ndarray, shape (K, n_features)
    """
    if K == 2:
        return np.asarray([[0., 0., 0.],
                           [0.07843137, 0.06666667, 0.09411765]])
    elif K == 4:
        return np.asarray([[0.72156863, 0.64313725, 0.54901961],
                           [0.49019608, 0.41960784, 0.33333333],
                           [0.02745098, 0., 0.],
                           [0.17254902, 0.16862745, 0.18823529]])
    elif K == 8:
        return np.asarray([[0.01568627, 0.01176471, 0.03529412],
                           [0.14509804, 0.12156863, 0.12941176],
                           [0.4745098, 0.40784314, 0.32941176],
                           [0.00784314, 0.00392157, 0.02745098],
                           [0.50588235, 0.43529412, 0.34117647],
                           [0.09411765, 0.09019608, 0.11372549],
                           [0.54509804, 0.45882353, 0.36470588],
                           [0.44705882, 0.37647059, 0.29019608]])
    elif K == 16:
        return np.asarray([[0.61568627, 0.56078431, 0.45882353],
                           [0.4745098, 0.38039216, 0.33333333],
                           [0.65882353, 0.57647059, 0.49411765],
                           [0.08235294, 0.07843137, 0.10196078],
                           [0.06666667, 0.03529412, 0.02352941],
                           [0.08235294, 0.07843137, 0.09803922],
                           [0.0745098, 0.07058824, 0.09411765],
                           [0.01960784, 0.01960784, 0.02745098],
                           [0.00784314, 0.00784314, 0.01568627],
                           [0.8627451, 0.78039216, 0.69803922],
                           [0.60784314, 0.52156863, 0.42745098],
                           [0.01960784, 0.01176471, 0.02352941],
                           [0.78431373, 0.69803922, 0.60392157],
                           [0.30196078, 0.21568627, 0.1254902],
                           [0.30588235, 0.2627451, 0.24705882],
                           [0.65490196, 0.61176471, 0.50196078]])
    else:
        print('This value of K is not supported.')
        return None


#classifing pixels to centroids
def find_closest_cen(centroids,X):
    centro_list= [[] for t in range(len(centroids))]
    for i in X:
        closest_cent = centroids[cdist([i], centroids).argmin()]
        cent_id = np.where(np.all(closest_cent == centroids, axis=1))[0][0]
        centro_list[cent_id].append(i)
    return centro_list

#updating centroids
def update_centroids(centro_list, centroids):
    for index in range(len(centroids)):
        pixels_num=len(centro_list[index])
        if pixels_num != 0:
            centroids[index] = sum(centro_list[index]) / pixels_num
        else:
            centroids[index] = 0
    return

# print centroids
def print_cent(cent):
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')
    else:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')[1:-1]

#loads data from inputed path
def load(path):
    A = imread(path)
    A_normalized = A.astype(float) / 255.
    return A_normalized

# data preperation (loading, normalizing, reshaping)
path = 'dog.jpeg'
centroids = []
k_list = [2, 4, 8, 16]
iter_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in k_list:
    if i != 2:
        print()
    centroids = init_centroids(None,i)
    A_normalized = load(path)
    img_size = A_normalized.shape
    X = A_normalized.reshape(img_size[0] * img_size[1], img_size[2])
    print(f'k={i}:')
    print('iter 0: ', end="")
    print(print_cent(centroids))
    #start learning
    for j in iter_list:
        centro_list = find_closest_cen(centroids, X)
        update_centroids(centro_list, centroids)
        print(f'iter {j}: ', end="")
        if j != 10:
            print(print_cent(centroids))
        else:
            print(print_cent(centroids), end="")

