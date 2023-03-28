"""
@description: Some helper functions to create gaussian heatmaps, computing Hu Moments etc
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


def gauss2d(img_size, mu, sigma, to_plot=False):
    w, h = img_size
    x = np.linspace(0, w, w)
    y = np.linspace(0, h, h)
    x, y = np.meshgrid(x, y)
    x_ = x.flatten()
    y_ = y.flatten()
    xy = np.vstack((x_, y_)).T
    normal_rv = multivariate_normal(mu, sigma)
    z = normal_rv.pdf(xy)
    cutoff_x = mu[0] - 3 * sigma[0][0]
    cutoff_y = mu[1] - 3 * sigma[1][1]
    cutoff_prob = normal_rv.pdf([cutoff_x, cutoff_y])
    z = z.reshape(w, h, order='F')
    z [z<cutoff_prob] = 0
    z = (z-np.mean(z))/np.std(z)
    if to_plot:
        plt.imsave("visualisations/guass2d.png", z)
    return z

def circle_map(img_size, center, radius, to_plot=False):
    w, h = img_size
    x = np.linspace(0, w, w, endpoint=False)
    y = np.linspace(0, h, h, endpoint=False)
    x, y = np.meshgrid(x, y)
    (h, k) = center
    z = (x-h)**2 + (y-k)**2
    z[z < radius**2] = 1
    z[z > radius**2] = 0
    if to_plot:
        plt.imsave("visualisations/circle_map.png", z)
    return z


def gauss_line(img_size, p1, p2, sigma=5, to_plot=False):
    w, h = img_size
    x = np.linspace(0, w, w)
    y = np.linspace(0, h, h)
    x, y = np.meshgrid(x, y)
    x_ = x.flatten()
    y_ = y.flatten()
    xy = np.vstack((x_, y_)).T
    a = get_distance_between_point_and_line(p1, p2, xy)
    normal_rv = multivariate_normal(0, sigma)
    z = normal_rv.pdf(a)
    cutoff = normal_rv.pdf(3*sigma)
    z [z<cutoff] = 0
    z = np.asarray(z, dtype=np.float32)
    z = z.reshape(w, h, order='F')
    dmin = int(min(p1[1], p2[1]))
    dmax = int(max(p1[1], p2[1]))
    z[:, dmax:-1] = 0
    z[:,:dmin] = 0
    z = (z-np.mean(z))/np.std(z)
    if to_plot:
        plt.imsave("visualisations/guass_line.png", z)
    return z

def rect_line(img_size, p1, p2, sigma=5, text=None, to_plot=False):
    w, h = img_size
    p1 = np.rint(p1)
    p2 = np.rint(p2)
    p1_outside = (p1[0] not in range(0, h)) or (p1[1] not in range(0, w))
    p2_outside = (p2[0] not in range(0, h)) or (p2[1] not in range(0, w))

    if p1_outside and p2_outside:
        z = np.zeros(img_size)
    else:
        x = np.linspace(0, w, w, endpoint=False)
        y = np.linspace(0, h, h, endpoint=False)
        x, y = np.meshgrid(x, y)
        x_ = x.flatten()
        y_ = y.flatten()
        xy = np.vstack((x_, y_)).T
        a = get_distance_between_point_and_line(p1, p2, xy)
        normal_rv = multivariate_normal(0, sigma)
        z = normal_rv.pdf(a)
        cutoff = normal_rv.pdf(3*sigma)
        z [z<cutoff] = 0
        z = np.asarray(z, dtype=np.float32)
        z = z.reshape(w, h, order='F')

        dmin_x = max(int(min(p1[1], p2[1])),0)
        dmax_x = min(int(max(p1[1], p2[1])), w-1)
        dmin_y = max(int(min(p1[0], p2[0])),0)
        dmax_y = min(int(max(p1[0], p2[0])), h-1)
        slope = np.abs(p1[0] - p2[0])/np.abs(p1[1] - p2[1] + np.finfo(dtype=np.float32).eps)
        if slope<1:
            z[:, dmax_x:] = 0
            z[:,:dmin_x] = 0
        else:
            z[dmax_y:, :] = 0
            z[:dmin_y, :] = 0
    
    if np.max(z) > 0:
        z = (z-np.mean(z))/np.std(z)
    else:
        z = z - 0.5
    if to_plot:
        plt.imsave("visualisations/guass_line.png", z)
    return z

# p1 and p2 specify line
def get_distance_between_point_and_line(p1,p2,p3):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
    return d

def get_angular_err(g1, g2, p1, p2):
    g = g1-g2
    m1 = g[1]/g[0]
    p = p1 - p2
    m2 = p[1]/p[0]
    theta = np.abs(np.arctan((m1-m2)/(1+m1*m2)))
    return theta

# raw moments
def M(i,j, I):
    w, h = I.shape
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    x, y = np.meshgrid(x, y)
    return np.sum((x**i)*(y**j)*I)

def get_centroid_or_argmax_of_2d_vector(vec, mode="centroid"):
    if mode == "centroid":
        centroid = np.array([(M(0,1, vec)/M(0,0, vec)), (M(1,0, vec)/M(0,0, vec))])
        return centroid
    elif mode == "argmax":
        return np.array(np.unravel_index(np.argmax(vec), vec.shape))

def orientation_angle_heatmap_line(vec):
    x_bar = M(0,1, vec)/M(0,0, vec)
    y_bar = M(1,0, vec)/M(0,0, vec)
    u20 = M(2,0, vec)/M(0,0, vec) - (y_bar**2)
    u02 = M(0,2, vec)/M(0,0, vec) - (x_bar**2)
    u11 = M(1,1, vec)/M(0,0, vec) - (x_bar*y_bar)
    theta = 0.5 * np.arctan(2*u11*(1/u20-u02))
    return theta

def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    ''' 
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    # Ref: https://www.jeremyjordan.me/semantic-segmentation/#loss
    '''
    
    # skip the batch for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape))) 
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    return 1 - np.mean((numerator + epsilon) / (denominator + epsilon))
