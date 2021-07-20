import numpy as np
import matplotlib.patches
import matplotlib.collections
import matplotlib.pyplot as plt
#plt.style.use('kostas')

def initial_xy_polar(r_min, r_max, r_N, theta_min, theta_max, theta_N):
    return np.array([[(r*np.cos(theta),r*np.sin(theta)) for r in np.linspace(r_min,r_max,r_N)] for theta in np.linspace(theta_min,theta_max,theta_N)])

def initial_xy_cartesian(x_min, x_max, x_N, y_min, y_max, y_N):
    return np.array([[(x,y) for x in np.linspace(x_min,x_max,x_N)] for y in np.linspace(y_min,y_max,y_N)])


def draw_footprint(A, axis_object=None, figure_object=None, axis=0, linewidth=4):
    '''
    Input A should be a 3-D numpy array with shape (Nx,Ny,2)
    representing a 2-D array of (x,y) points. This function
    will draw lines between adjacent points in the 2-D array.
    '''
    if len(A.shape) != 3:
        print('ERROR: Invalid input matrix')
        return None
    if A.shape[2] != 2:
        print('ERROR: Points are not defined in 2D space')
        return None

    sx = A.shape[0]-1
    sy = A.shape[1]-1

    p1 = A[:-1,:-1,:].reshape(sx*sy,2)[:,:]
    p2 = A[1:,:-1,:].reshape(sx*sy,2)[:]
    p3 = A[1:,1:,:].reshape(sx*sy,2)[:]
    p4 = A[:-1,1:,:].reshape(sx*sy,2)[:]

    #Stack endpoints to form polygons
    Polygons = np.stack((p1,p2,p3,p4))
    #transpose polygons
    Polygons = np.transpose(Polygons,(1,0,2))
    patches = list(map(matplotlib.patches.Polygon,Polygons))

    #assign colors
    patch_colors = [(0,0,0) for a in Polygons]
    patch_colors[(sx-1)*sy:] = [(0,1,0)]*sy
    patch_colors[(sy-1)::sy] = [(0,0,1)]*sx

    p_collection = matplotlib.collections.PatchCollection(patches,facecolors=[],linewidth=linewidth,edgecolor=patch_colors)

    if axis_object is None:
        if figure_object:
            fig = figure_object
        else:
            fig = plt.figure()
        if len(fig.axes) == 0:
            plt.subplot(1,1,1)
        if axis >= len(fig.axes) or axis < 0:
            i = 0
        else:
            i = axis
        ax = fig.axes[i]
    else:
        ax = axis_object
        fig = None

    ax.add_collection(p_collection)

    return fig

def example():
    coords_matrix = initial_xy_polar(0.01,5,11,0.,np.pi/2.,10)
    fig = draw_footprint(coords_matrix)
    fig.axes[0].set_xlim(-0.05,5.1)
    fig.axes[0].set_ylim(-0.05,5.1)
