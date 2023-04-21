import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import cv2


# 卷积
def convolve(filter, mat, padding, strides):
    result = None
    filter_size = filter.shape
    mat_size = mat.shape
    if len(filter_size) == 2:
        if len(mat_size) == 3:
            channel = []
            for i in range(mat_size[-1]):
                pad_mat = np.pad(mat[:, :, i], ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
                temp = []
                for j in range(0, mat_size[0], strides[1]):
                    temp.append([])
                    for k in range(0, mat_size[1], strides[0]):
                        value = (filter * pad_mat[j:j + filter_size[0], k:k + filter_size[1]]).sum()
                        temp[-1].append(value)
                channel.append(np.array(temp))
            channel = tuple(channel)
            result = np.dstack(channel)
        elif len(mat_size) == 2:
            channel = []
            pad_mat = np.pad(mat[:, :], ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
            for j in range(0, mat_size[0], strides[1]):
                channel.append([])
                for k in range(0, mat_size[1], strides[0]):
                    value = (filter * pad_mat[j:j + filter_size[0], k:k + filter_size[1]]).sum()
                    channel[-1].append(value)
            result = np.array(channel)
    return result


# 下采样
def downsample(img, step):
    img = img[::step, ::step]
    print(img.shape)
    return img


# 高斯核
def gaussianKernel(sigma, kernel_size):
    temp = [t - (kernel_size // 2) for t in range(kernel_size)]
    assistant = []
    for i in range(kernel_size):
        assistant.append(temp)
    assistant = np.array(assistant)
    temp = sigma * sigma * 2
    result = (1.0 / (temp * np.pi)) * np.exp(-(assistant ** 2 + (assistant.T) ** 2) / temp)
    result = result / np.sum(result)
    return result


# 高斯差分金字塔
def getDoG(img, n, sigma0, Octave=None, S=None):
    if S == None:
        S = n + 3
    if Octave == None:
        Octave = int(np.log2(min(img.shape[0], img.shape[1]))) - 3
    sigma = [[sigma0 * (2 ** ((s / n) + o)) for s in range(S)] for o in range(Octave)]
    samplePyramid = [downsample(img, 2 ** o) for o in range(Octave)]

    guassianPyramid = []
    for i in range(Octave):
        guassianPyramid.append([])
        for j in range(S):
            kernel_size = int(6 * sigma[i][j] + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            guassianPyramid[-1].append(convolve(gaussianKernel(sigma[i][j], kernel_size), samplePyramid[i],
                                                [kernel_size // 2, kernel_size // 2, kernel_size // 2,
                                                 kernel_size // 2], [1, 1]))

    DoG = [[guassianPyramid[o][s + 1] - guassianPyramid[o][s] for s in range(S - 1)] for o in range(Octave)]
    # for o in range(Octave):
    #     for s in range(S):
    #         cv2.imwrite("./result/O/" + "/" + str(o) + "--" + str(s) + ".jpg", guassianPyramid[o][s])
    #         print(str(o), str(s), guassianPyramid[o][s].shape)
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    # for o in range(Octave):
    #     for s in range(S - 1):
    #         cv2.imwrite("./result/DoG/" + "/" + str(o) + "--" + str(s) + ".jpg", DoG[o][s])
    #         print(str(o), str(s), DoG[o][s].shape)
    return DoG, guassianPyramid


# 亚像素精准定位
def localAdjust(DoG, o, s, x, y, contrastThreshold, edgfThreshold, sigma, n):
    sift_max_interp_steps = 5
    sift_img_border = 5
    point = []
    img_scale = 1.0 / 255.0
    derive_scale = img_scale * 0.5
    second_derive_scale = img_scale
    cross_derive_scale = img_scale * 0.25

    img = DoG[o][s]
    i = 0
    while i < sift_max_interp_steps:
        if s < 1 or s > n or y < sift_img_border or y >= img.shape[1] - sift_img_border or x < sift_img_border or x >= \
                img.shape[0] - sift_img_border:
            return None, None, None, None
        img = DoG[o][s]
        prev = DoG[o][s - 1]
        next = DoG[o][s + 1]

        dD = [(img[x, y + 1] - img[x, y - 1]) * derive_scale, (img[x + 1, y] - img[x - 1, y]) * derive_scale,
              (next[x, y] - prev[x, y]) * derive_scale]

        dxx = (img[x, y + 1] + img[x, y - 1] - (2 * img[x, y])) * second_derive_scale
        dyy = (img[x + 1, y] + img[x - 1, y] - 2 * img[x, y]) * second_derive_scale
        dss = (next[x, y] + prev[x, y] - 2 * img[x, y]) * second_derive_scale
        dxy = ((img[x - 1, y - 1] + img[x + 1, y + 1]) - (img[x + 1, y - 1] + img[x - 1, y + 1])) * cross_derive_scale
        dxs = ((next[x, y + 1] + prev[x, y - 1]) - (prev[x, y + 1] + next[x, y - 1])) * cross_derive_scale
        dys = ((next[x + 1, y] + prev[x - 1, y]) - (prev[x + 1, y] + next[x - 1, y])) * cross_derive_scale

        H = [[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]]

        X = np.matmul(np.linalg.pinv(np.array(H)), np.array(dD))

        xi = -X[2]
        xr = -X[1]
        xc = -X[0]

        if np.abs(xi) < 0.5 and np.abs(xr) < 0.5 and np.abs(xc) < 0.5:
            break
        y += int(np.round(xc))
        x += int(np.round(xr))
        s += int(np.round(xi))

        i += 1
    # 迭代次数
    if i >= sift_max_interp_steps:
        return None, x, y, s
    if s < 1 or s > n or y < sift_img_border or x < sift_img_border or y >= img.shape[1] - sift_img_border or x >= \
            img.shape[0] - sift_img_border:
        return None, None, None, None
    # 极值点下的极值
    f_X = img[x, y] * img_scale + ((np.array(dD)).dot(np.array([xc, xr, xi])) / 2)
    # 舍去对比度低的点
    if np.abs(f_X) < contrastThreshold / n:
        return None, x, y, s

    tr_H = dxx + dyy
    det_H = dxx * dyy - dxy * dxy
    if det_H <= 0 or tr_H * tr_H / det_H >= (edgfThreshold + 1) ** 2 / edgfThreshold:
        return None, x, y, s
    point.append((x + xr) * (2 ** o))
    point.append((y + xc) * (2 ** o))
    point.append(o + (s << 8))
    # point.append(o+(s<<8)+(int(np.round((xi+0.5)*255))<<16))
    point.append(sigma * (2 ** ((s + xi) / n + o + 1)))

    return point, x, y, s


# 方向
def GetMainDirection(img, r, c, radius, sigma, Binnum):
    expf_scale = -1.0 / (2.0 * sigma * sigma)
    X = []
    Y = []
    W = []
    tempHist = []
    for i in range(Binnum):
        tempHist.append(0.0)
    k = 0
    for i in range(-radius, radius + 1):
        y = r + i
        if y <= 0 or y >= img.shape[0] - 1:
            continue
        for j in range(-radius, radius + 1):
            x = c + j
            if x <= 0 or x >= img.shape[1] - 1:
                continue
            dx = (img[y, x + 1] - img[y, x - 1])
            dy = (img[y - 1, x] - img[y + 1, x])

            X.append(dx)
            Y.append(dy)
            W.append((i * i + j * j) * expf_scale)
            k += 1
    length = k
    W = np.exp(np.array(W))
    Y = np.array(Y)
    X = np.array(X)
    Ori = np.arctan2(Y, X) * 180 / np.pi
    Mag = np.sqrt(X * X + Y * Y)

    for k in range(length):
        bin = int(np.round((Binnum / 360) * Ori[k]))
        if bin >= Binnum:
            bin -= Binnum
        if bin < 0:
            bin += Binnum
        tempHist[bin] += W[k] * Mag[k]

    temp = [tempHist[Binnum - 1], tempHist[Binnum - 2], tempHist[0], tempHist[1]]
    tempHist.insert(0, temp[0])
    tempHist.insert(0, temp[1])
    tempHist.insert(len(tempHist), temp[2])
    tempHist.insert(len(tempHist), temp[3])
    hist = []
    for i in range(Binnum):
        hist.append((tempHist[i] + tempHist[i + 4]) * (1 / 16) + (tempHist[i + 1] + tempHist[i + 3]) * (4 / 16) + (
                    tempHist[i + 2] * (6 / 16)))
    maxval = max(hist)
    return maxval, hist


# 极值点
def findPoint(DoG, sigma, guassianPyramid, n, Binnum=36, contrastThreshold=0.04, edgfThreshold=10.0):
    sigma0 = 1.52
    radius = 3 * sigma0
    peakRatio = 0.8
    keypoint = []
    for o in range(len(DoG)):
        for s in range(1, len(DoG[0]) - 1):
            threshold = 0.5 * contrastThreshold / (n * 255)
            for i in range(DoG[o][s].shape[0]):
                for j in range(DoG[o][s].shape[1]):
                    val = DoG[o][s][i, j]
                    eight_neiborhood_prev = DoG[o][s - 1][max(0, i - 1):min(i + 2, DoG[o][s - 1].shape[0]),
                                            max(0, j - 1):min(j + 2, DoG[o][s - 1].shape[1])]
                    eight_neiborhood = DoG[o][s][max(0, i - 1):min(i + 2, DoG[o][s].shape[0]),
                                       max(0, j - 1):min(j + 2, DoG[o][s].shape[1])]
                    eight_neiborhood_next = DoG[o][s + 1][max(0, i - 1):min(i + 2, DoG[o][s + 1].shape[0]),
                                            max(0, j - 1):min(j + 2, DoG[o][s + 1].shape[1])]
                    if np.abs(val) > threshold and ((val > 0 and (val >= eight_neiborhood_prev).all() and (
                            val >= eight_neiborhood).all() and (val >= eight_neiborhood_next).all()) or (
                                                            val < 0 and (val <= eight_neiborhood_prev).all() and (
                                                            val >= eight_neiborhood).all() and (
                                                                    val >= eight_neiborhood_next).all())):
                        point, x, y, layer = localAdjust(DoG, o, s, i, j, contrastThreshold, edgfThreshold, sigma, n)
                        if point == None:
                            continue
                        scl_octave = point[-1] / (2 ** (o + 1))
                        omax, hist = GetMainDirection(guassianPyramid[0][layer], x, y,
                                                      int(np.round(radius * scl_octave)), sigma0 * scl_octave, Binnum)
                        mag_thr = omax * peakRatio
                        for k in range(Binnum):
                            if k > 0:
                                l = k - 1
                            else:
                                l = Binnum - 1
                            if k < Binnum - 1:
                                r2 = k + 1
                            else:
                                r2 = 0
                            if hist[k] > hist[l] and hist[k] > hist[r2] and hist[k] >= mag_thr:
                                bin = k + 0.5 * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[k] + hist[r2])
                                if bin < 0:
                                    bin = Binnum + bin
                                else:
                                    bin = bin - Binnum
                                temp = point[:]
                                temp.append((360 / Binnum) * bin)
                                keypoint.append(temp)
    return keypoint


def calcSIFT_Descriptors(img, ptf, ori, scl, d, n, SIFT_DESCR_SCL_FCTR=3.0, mag=0.2, SIFT_INT_DESCR_FCTR=512.0,
                         EPSILON=1.1920920E-07):
    dst = []
    pt = [int(np.round(ptf[0])), int(np.round(ptf[1]))]
    cost = np.cos(ori * np.pi / 180)
    sint = np.sin(ori * np.pi / 180)
    bins_per_rad = n / 360.0
    exp_scale = -1.0 / (d * d * 0.5)
    hist_width = SIFT_DESCR_SCL_FCTR * scl
    radius = int(np.round(hist_width * (np.sqrt(2)) * (d + 1) * 0.5))
    cost /= hist_width
    sint /= hist_width
    rows = img.shape[0]
    cols = img.shape[1]

    hist = [0.0] * ((d + 2) * (d + 2) * (n + 2))
    X = []
    Y = []
    rBin = []
    cBin = []
    W = []

    k = 0
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            c_rot = j * cost - i * sint
            r_rot = j * sint + i * cost
            rbin = r_rot + d // 2 - 0.5
            cbin = c_rot + d // 2 - 0.5
            r = pt[1] + i
            c = pt[1] + i

            if rbin > -1 and rbin < d and cbin > -1 and cbin < d and r > 0 and r < rows - 1 and c > 0 and c < cols - 1:
                dx = (img[r, c + 1] - img[r, c - 1])
                dy = (img[r - 1, c] - img[r + 1, c])
                X.append(dx)
                Y.append(dy)
                rBin.append(rbin)
                cBin.append(cbin)
                W.append((r_rot ** 2 + c_rot ** 2) * exp_scale)
                k += 1

    length = k
    Y = np.array(Y)
    X = np.array(X)
    Ori = np.arctan2(Y, X) * 180 / np.pi
    Mag = (X ** 2 + Y ** 2) ** 0.5
    W = np.exp(np.array(W))

    for k in range(length):
        rbin = rBin[k]
        cbin = cBin[k]
        obin = (Ori[k] - ori) * bins_per_rad
        mag = Mag[k] * W[k]
        r0 = int(rbin)
        c0 = int(cbin)
        o0 = int(obin)
        rbin -= r0
        cbin -= c0
        obin -= o0
        if o0 < 0:
            o0 += n
        if o0 >= n:
            o0 -= n

        # 三线性插值
        v_r1 = mag * rbin
        v_r0 = mag - v_r1

        v_rc11 = v_r1 * cbin
        v_rc10 = v_r1 - v_rc11
        v_rc01 = v_r0 * cbin
        v_rc00 = v_r0 - v_rc01

        v_rco111 = v_rc11 * obin
        v_rco110 = v_rc11 - v_rco111
        v_rco101 = v_rc10 * obin
        v_rco100 = v_rc10 - v_rco101
        v_rco011 = v_rc01 * obin
        v_rco010 = v_rc01 - v_rco011
        v_rco001 = v_rc00 * obin
        v_rco000 = v_rc00 - v_rco001

        idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0
        hist[idx] += v_rco000
        hist[idx + 1] += v_rco001
        hist[idx + n + 2] += v_rco010
        hist[idx + n + 3] += v_rco011
        hist[idx + (d + 2) * (n + 2)] += v_rco100
        hist[idx + (d + 2) * (n + 2) + 1] += v_rco101
        hist[idx + (d + 3) * (n + 2)] += v_rco110
        hist[idx + (d + 3) * (n + 2) + 1] += v_rco111

    for i in range(d):
        for j in range(d):
            idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2)
            hist[idx] += hist[idx + n]
            hist[idx + 1] += hist[idx + n + 1]
            for k in range(n):
                dst.append(hist[idx + k])

    nrm2 = 0
    length = d * d * n
    for k in range(length):
        nrm2 += dst[k] * dst[k]
    thr = np.sqrt(nrm2) * mag

    nrm2 = 0
    for i in range(length):
        val = min(dst[i], thr)
        dst[i] = val
        nrm2 += val * val
    nrm2 = SIFT_INT_DESCR_FCTR / max(np.sqrt(nrm2), EPSILON)
    for k in range(length):
        dst[k] = min(max(dst[k] * nrm2, 0), 255)
    return dst


# 计算描述符
def calcDescriptors(guassianPyramid, keypoints, SIFT_DESCR_WIDTH=4, SIFT_DESCR_HIST_BINS=8):
    d = SIFT_DESCR_WIDTH
    n = SIFT_DESCR_HIST_BINS
    descriptors = []
    for i in range(len(keypoints)):
        kpt = keypoints[i]
        o = kpt[2] & 255
        s = (kpt[2] >> 8) & 255
        scale = 1.0 / (1 << o)
        size = kpt[3] * scale
        ptf = [kpt[1] * scale, kpt[0] * scale]
        img = guassianPyramid[o][s]

        descriptors.append(calcSIFT_Descriptors(img, ptf, kpt[-1], size * 0.5, d, n))
    return descriptors


# sift特征匹配
def SIFT(img, showDoGimgs=False):
    SIFT_SIGMA = 1.6
    SIFT_INIT_SIGMA = 0.5
    sigma0 = (SIFT_SIGMA ** 2 - SIFT_INIT_SIGMA ** 2) ** 0.5
    n = 3
    DoG, GuassianPyramid = getDoG(img, n, sigma0)
    if showDoGimgs:
        for i in DoG:
            for j in i:
                plt.imshow(j.astype(np.uint8), cmap='gray')
                plt.axis('off')
                plt.show()
    KeyPoints = findPoint(DoG, SIFT_SIGMA, GuassianPyramid, n)
    discriptors = calcDescriptors(GuassianPyramid, KeyPoints)
    return KeyPoints, discriptors


# 划线
def Lines(img,info,color = (255,215,0),err = 250):
    if len(img.shape) == 2:
        result = np.dstack((img,img,img))
    else:
        result = img
    k = 0
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            temp = (info[:,1]-info[:,0])
            A = (j - info[:,0])*(info[:,3]-info[:,2])
            B = (i - info[:,2])*(info[:,1]-info[:,0])
            temp[temp==0]=1e-9
            t=(j-info[:,0])/temp
            e=np.abs(A-B)
            temp=e < err
            if (temp*(t>=0)*(t<=1)).any():
                result[i,j]=color
                k+=1
    print(k)
    return result
def drawLine(X1,X2,Y1,Y2,dis,img,num=10):
    info = list(np.dstack((X1,X2,Y1,Y2,dis))[0])
    info=sorted(info,key=lambda x:x[-1])
    info=np.array(info)
    info=info[:min(num,info.shape[0]),:]
    img=Lines(img,info)
    plt.imsave('./result/3.jpg',img)
    if len(img.shape)==2:
        plt.imshow(img.astype(np.uint8),cmap='gray')
    else:
        plt.imshow(img.astype(np.uint8))
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    origimg = plt.imread('./imgs/h.jpg')
    if len(origimg.shape) == 3:
        img = origimg.mean(axis=-1)
    else:
        img = origimg
    keyPoints,discriptors = SIFT(img)


    origimg2 = plt.imread('./imgs/hh.jpg')
    if len(origimg2.shape) == 3:
        img2 = origimg2.mean(axis=-1)
    else:
        img2 = origimg2
    ScaleRatio = img.shape[0]*1.0/img2.shape[0]

    img2 = np.array(Image.fromarray(img2).resize((int(round(ScaleRatio * img2.shape[1])),img.shape[0]), Image.BICUBIC))
    keyPoints2, discriptors2 = SIFT(img2)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(discriptors,[0]*len(discriptors))
    match = knn.kneighbors(discriptors2,n_neighbors=1,return_distance=True)

    keyPoints = np.array(keyPoints)[:,:2]
    keyPoints2 = np.array(keyPoints2)[:,:2]

    keyPoints2[:, 1] = img.shape[1] + keyPoints2[:, 1]

    origimg2 = np.array(Image.fromarray(origimg2).resize((img2.shape[1],img2.shape[0]), Image.BICUBIC))
    result = np.hstack((origimg,origimg2))


    keyPoints = keyPoints[match[1][:,0]]

    X1 = keyPoints[:, 1]
    X2 = keyPoints2[:, 1]
    Y1 = keyPoints[:, 0]
    Y2 = keyPoints2[:, 0]

    drawLine(X1,X2,Y1,Y2,match[0][:,0],result)