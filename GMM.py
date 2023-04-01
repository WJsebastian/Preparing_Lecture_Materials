import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import time
row = 0
column = 0
K = 0

def initialize():

    global mean, weight, deviation
    mean = np.zeros(shape=(K, row, column, 3))
    weight = np.ones(shape=(K, row, column)) / K
    deviation = np.ones(shape=(K, row, column))
    
def check():

    global mask_divide, idx
    T = 0.7
    ratio = -1*(weight / deviation)
    idx = ratio.argsort(axis=0)
    ratio.sort(axis=0)
    ratio *= -1
    cum = np.cumsum(ratio, axis=0)
    mask_divide = (cum < T)
    mask_divide = np.choose(idx, mask_divide)
    # mask_divide = np.choose(idx, mask_divide)

def mahalanobis_probability(video):

    global mask_distance, prob

    temp = np.subtract(video, mean)
    temp = np.sum(temp**2, axis=3) / (deviation**2)

    prob = np.exp(temp/(-2)) / (np.sqrt((2*np.pi)**3)*deviation)

    temp = np.sqrt(temp)
    mask_distance = (temp < 2.5*deviation)

def update(video):
    
    global weight, mask_distance, mean, deviation, prob, mask_some

    alpha = 0.2
    rho = alpha * prob
    
    # print(mask_distance)
    
    mask_some = np.bitwise_or.reduce(mask_distance, axis=0)
    # print(mask_some)
    mask_update = np.where(mask_some == True, mask_distance, -1)

    weight = np.where(mask_update == 1, (1 - alpha) * weight + alpha, weight)
    weight = np.where(mask_update == 0, (1 - alpha) * weight, weight)
    #weight = np.where(mask_update == -1, 0.0001, weight)

    data = np.stack([video]*K, axis=0)
    mask = np.stack([mask_update]*3, axis=3)
    r = np.stack([rho]*3, axis=3)
    
    mean = np.where(mask == 1, (1 - r) * mean + r * data, mean)
    mean = np.where(mask == -1, data, mean)
    
    deviation = np.where(mask_update == 1, np.sqrt((1-rho)*(deviation**2) + rho*(np.sum(np.subtract(video, mean)**2, axis=3))), deviation)
    deviation = np.where(mask_update == -1, 3 + np.ones(shape=(K, row, column)), deviation)


def result(video):

    background = np.zeros(shape=(row, column, 3), dtype=np.uint8)
    foreground = 255 + np.zeros(shape=(row, column, 3), dtype=np.uint8)
    m = np.stack([mask_some]*3, axis=2)
    res1 = np.where(m == False, foreground, background)
    # cv2.imshow('res1', res1)
    n = np.bitwise_and(mask_divide, mask_distance)
    n = np.bitwise_or.reduce(n, axis=0)
    n = np.stack([n]*3, axis=2)
    res = np.where(n == True, background, foreground)

    # cv2.imshow('frame', video)
    # cv2.imshow('res', res)
    # print(res)
    res = np.bitwise_or(res1, res)
    k1 = np.bitwise_or.reduce(mask_distance, axis=0)
    k1 = np.stack([k1]*3, axis=2)
    k1 = np.where(k1== True, foreground, background)

    k2 = np.bitwise_or.reduce(mask_divide, axis=0)
    k2 = np.stack([k2]*3, axis=2)
    k2 = np.where(k2== True, foreground, background)
    # cv2.imshow('mask_distance', k1)
    # cv2.imshow('mask_divide', k2)
    # cv2.imshow('m',res1)
    return res

def frame_processing(video):
    
    check()
    mahalanobis_probability(video)
    update(video)
    return result(video)

def main():

    global row, column, K
    cap = cv2.VideoCapture('2.mov')
    #cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    K = 5
    row, column, _ = frame.shape
    initialize()
    # fgbg = cv2.createBackgroundSubtractorMOG2(0)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    output = cv2.VideoWriter('output_video.mp4', fourcc, 25, (2*column,row))
    ret, frame = cap.read()
    if ret == True:
        frame = frame_processing(frame)
    cnt = 1
    while(1):
        ret, frame = cap.read()
        if ret == True:
            #frame = 128 + np.zeros(shape=(row, column, 3), dtype=np.uint8)
            res = frame_processing(frame)
            # cv2.imshow('frame', frame)
            # fgmask = fgbg.apply(frame)
            # cv2.imshow('fgmask', fgmask)
            BG = cv2.resize(frame, (column * 2, row), interpolation= cv2.INTER_AREA)
            BG[0:row, 0:column] = frame
            BG[0:row, column:2*column] = res
            # if cnt % 2 ==0:
            cv2.imshow('video', BG) 
                # output.write(BG)
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
        # time.sleep(0.01)
        cnt +=1

    cap.release()
    output.release()
    cv2.destroyAllWindows()

main()

