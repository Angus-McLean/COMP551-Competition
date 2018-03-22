import numpy as np
from scipy.ndimage import median_filter
from skimage.transform import resize
import matplotlib.pyplot as plt


def loadData(filename):
    return np.loadtxt(filename, delimiter=",").reshape(-1, 64, 64)

def toBinary(img, threshold):
    # Filter just dark colors
    binX = np.floor_divide(img, threshold)
    return binX

def pixelCoords(img):
    # group and get bounds of hand written digits
    return zip(*np.nonzero(img))

def connectedCluster(img, distance=4):
    groups = []

    binCoords = pixelCoords(img)

    for pt in binCoords:
        # distances = array of distances from each group for the current point
        distances = np.array([
            np.absolute(np.subtract(grp, pt)).sum(axis=1).min()
            for grp in groups
        ])

        # which of these currentPt<-->group distances are within the grouping threshold
        withInPixDist = np.where(distances < distance)[0]
        if(len(withInPixDist)==0):
            # no groups are within pixel distance
            groups.append([pt])
        else:
            groups[np.argmin(distances)].append(pt)
            # handle grouping groups
            if len(withInPixDist) > 1:
                # current point joins 2 groups
                groups[withInPixDist[0]] = groups[withInPixDist[0]] + groups[withInPixDist[1]]
    return groups

def getBounds(group):
    mins = np.min(group, axis=0)
    maxs = np.max(group, axis=0)
    return {
        'x' : (mins[1], maxs[1], maxs[1] - mins[1]),
        'y' : (mins[0], maxs[0], maxs[0] - mins[0])
    }

def largestBounds(boundsArr):
    indLargetImg = np.argmax([max(bound['x'][2], bound['y'][2]) for bound in boundsArr])
    maxBound = boundsArr[indLargetImg]
    return maxBound


def sliceImg(img, bounds):
    return img[bounds['y'][0]:bounds['y'][1], bounds['x'][0]:bounds['x'][1]]

# places an image slice centered inside a target shape and returns the full image
# params : (2d np array)
def cropCenter(img, targetShape):
    img = getImgCenter(img, (min(img.shape[0], targetShape[0]), min(img.shape[1], targetShape[1])))
    imgFinal = np.zeros(targetShape)
    imgFinalCenter = getImgCenter(imgFinal, img.shape)
    imgFinalCenter[:,:] = img
    return imgFinal

# given an image (2d np array)
# params : (2d-np-array, (target_height, target_width))
def getImgCenter(img, targetShape):
    halfSlice = (img.shape[0]//2, img.shape[1]//2)
    halfTarg = (targetShape[0]//2, targetShape[1]//2)

    bottomBnd = (halfSlice[0] - (halfTarg[0]), halfSlice[1] - (halfTarg[1]))
    topBnd = (halfSlice[0] + (targetShape[0]-halfTarg[0]), halfSlice[1] + (targetShape[1]-halfTarg[1]))
    return img[bottomBnd[0]:topBnd[0], bottomBnd[1]:topBnd[1]]

def scaledStretch(img, targetShape):
    scale0 = float(targetShape[0])/img.shape[0]
    scale1 = float(targetShape[1])/img.shape[1]

    scaledImg = resizeImage(img, min(scale0, scale1))

    imgFinal = np.zeros(targetShape)
    imgFinalCenter = getImgCenter(imgFinal, scaledImg.shape)
    imgFinalCenter[:,:] = scaledImg
    return imgFinal

def roundImageBin(img, window_size):
    rndImgBin = median_filter(img, size=(window_size,window_size))
    return rndImgBin

def resizeImage(img, factor):
    tmpImg = resize(img, (int(img.shape[0] * factor), int(img.shape[1] * factor)))
    return np.floor_divide(tmpImg, 1)

def displayImgs(imgs, titles=[], n_cols=4):
    numRows = int(np.ceil(len(imgs)/float(n_cols)))
    f, axarr = plt.subplots(numRows, n_cols)
    f.subplots_adjust(hspace=0.5)
    for i in range(numRows):
        for j in range(n_cols):
            row = axarr if numRows == 1 else axarr[i]
            if i*4+j < len(imgs):
                row[j].set_title(titles[i*n_cols+j] if i*n_cols+j in titles else "")
                row[j].imshow(imgs[i*n_cols+j])
