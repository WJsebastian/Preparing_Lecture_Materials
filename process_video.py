from gmm_v2 import GMM
import cv2
import torch
cap = cv2.VideoCapture('2.mov')
#cap = cv2.VideoCapture(0)
ret, frame = cap.read()
K = 3
row, column, _ = frame.shape
gmm = GMM(frame, n_components=K)
# fgbg = cv2.createBackgroundSubtractorMOG2(0)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
# output = cv2.VideoWriter('output_video.mp4', fourcc, 25, (2*column,row))

while True:
    
    if ret == True:
        frame_shown = gmm.process(frame)
        BG = cv2.resize(frame, (column * 2, row), interpolation= cv2.INTER_AREA)
        BG[0:row, 0:column] = frame
        BG[0:row, column:2*column] = frame_shown
        cv2.imshow('video', BG) 
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    else :
        break
    ret, frame = cap.read()
    # time.sleep(0.01)

cap.release()
# output.release()
cv2.destroyAllWindows()