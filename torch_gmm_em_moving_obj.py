import cv2
import torch
import torchvision.transforms as transforms
class GMM(torch.nn.Module):
    def __init__(self, frames, n_components = 3, T = 0.7, eps = 1.3-6):
        super(GMM, self).__init__()
        self.n_components = n_components
        self.h, self.w, _ = frames[0].shape
        self.mean = torch.zeros(self.n_components, self.h, self.w, 3) #(K, h, w, 3)
        self.deviation = torch.ones(self.n_components, self.h, self.w) # (K, h, w)
        self.weight = 1 / self.n_components * torch.ones(self.n_components,self.h, self.w) #(K, h, w)
        self.transform  = transforms.ToTensor()
        self.histogram = []
        self.N = len(frames)
        self.T = T
       
        self.ratio = torch.zeros(self.n_components, self.h, self.w) #(k, h, w)
        # self.foreground_choice = torch.zeros(self.n_components, self.h, self.w, dtype=torch.bool) #(K, h, w)
        self.foreground_mask = torch.zeros(self.n_components, self.h, self.w, dtype = torch.bool)
        self.eps = eps
        self.n_iter = 100
        self.fit(frames)

        
    # def prob(self, frame_tensor):
    #     frame_tensor = frame_tensor.transpose(1, 2, 0).unsqueeze(0)
    #     return 1 / (2* torch.pi * self.deviation) *\
    #           torch.exp(- 1/ 2 *torch.square(frame_tensor - self.mean) \
    #                     / torch.square(self.deviation))
    def prob(self, X):
        # X: (N, h, w, 3)
        X = X.unsqueeze(1) #(N, 1, h, w, 3)
        deviation = self.deviation.unsqueeze(0) #(1, K, h, w)
        mean = self.mean.unsqueeze(0) #(1, K, h, w, 3)
        temp = torch.sum(torch.square(X - mean), dim=-1) / torch.square(deviation)
        return torch.exp(-temp / 2) / (torch.sqrt(torch.tensor((2 * torch.pi) ** 3)) * deviation)
    
    def fit(self, frames):
        for frame in frames:
            frame_tensor = self.transform(frame).permute(1, 2, 0)
            self.histogram.append(frame_tensor)
            
        X = torch.stack(self.histogram) # (N, h, w, 3)
        # R = self.weight * self.prob(frame) / torch.sum(self.weight * self.prob(frame), dim=0).unsqueeze(0) #(K, h, w, 3)
        ## Important step, because we need to introduce randomness
        # tensor_int = (255 * X).to(torch.uint8) #(N, h, w, 3)
        # intensity , counts = tensor_int.unique(return_counts=True, dim=0) #(N', h, w, 3)
        # count_k, index = counts.topk(self.n_components, dim=0) #(K, h, w, 3)
        # self.mean = 1.0 * torch.gather(intensity, dim=0, index=index) / 255.0
        # self.deviation = 3 * self.N / count_k
        self.mean[0,:,:,:] = 1/4
        self.mean[1,:,:,:] = 2/4
        self.mean[2,:,:,:] = 3/4
        iter = 0
        error = torch.inf
        log_likelihood_old = torch.log(torch.sum(self.prob(X), dim=0))#(h, w)

        ## GMM
        while iter < self.n_iter and error > self.eps:
            print(iter)
            Prob = self.prob(X) # (N, K, h, w)

            

            R_0 = self.weight.unsqueeze(0) * Prob #(N, K, h, w)
            R = R_0 / R_0.sum(dim=1).unsqueeze(1) #(N, K, h, w)
            # print(R)
            ## M Step
            m = R.shape[0]
            self.weight = R.sum(dim = 0) / m #(K, h, w)

            self.mean = torch.sum(R.unsqueeze(-1) * X.unsqueeze(1), dim=0) / R.sum(dim=0).unsqueeze(-1) #(K, H, W, 3)
            self.deviation = torch.sum(R * torch.sum(torch.square(X.unsqueeze(1) \
                                                                - self.mean.unsqueeze(0)), dim=-1), dim=0) / R.sum(dim=0) #(k, h, w)
            log_likelihood = torch.log(torch.sum(self.prob(X), dim=0)) #(h, w)
            error = torch.max(torch.abs(log_likelihood - log_likelihood_old))
            log_likelihood_old = log_likelihood
        self.ratio = self.weight / self.deviation #(K, h, w)
        # print(self.ratio)
        # values, idx = torch.sort(self.ratio, dim=0)
        # self.mean = torch.gather(self.mean, dim=0, index=idx.unsqueeze(-1).repeat(1, 1, 1, 3))
        # self.deviation = torch.gather(self.deviation, dim=0, index=idx)
        # print(self.deviation.shape)
        # self.weight = torch.gather(self.weight, dim=0, index=idx)
        

        # cum = torch.cumsum(values, dim=0) #(K, h, w)
        # self.foreground_choice[cum < self.T] = True
        # foreground_choice = (cum < self.T)
        # print(foreground_choice.shape)
        # print(cum)
        # self.foreground_mask = cum < self.T
        # print(self.ratio.shape)
        # print(self.ratio.min(dim=1).values.shape)
        # self.foreground_mask = self.ratio >= self.ratio.max(dim=0).values
        self.foreground_mask = self.ratio <= self.ratio.min(dim=0).values
        # print(self.foreground_mask.shape)
        # print(self.foreground_mask)

    def update(self, frame):
        ## E Step
        if len(self.histogram) == self.N:
            return
        frame_tensor = self.transform(frame)
        frame_tensor = frame_tensor.permute(1, 2, 0)
        self.histogram.append(frame_tensor)
        frame_tensor = frame_tensor.unsqueeze(0)
        X = torch.stack(self.histogram) # (N, h, w, 3)
        R = self.weight * self.prob(frame) / torch.sum(self.weight * self.prob(frame), dim=0).unsqueeze(0) #(K, h, w, 3)
        Prob = self.prob(X) # (N, K, h, w)

        print(Prob)
        R_0 = self.weight.unsqueeze(0) * Prob #(N, K, h, w)
        R = R_0 / R_0.sum(dim=1).unsqueeze(1) #(N, K, h, w)


        ## M Step
        m = R.shape[0]
        self.weight = R.sum(dim = 0) / m #(K, h, w)
       

        self.mean = torch.sum(R.unsqueeze(-1) * X.unsqueeze(1), dim=0) / R.sum(dim=0).unsqueeze(-1) #(K, H, W, 3)
        self.deviation = torch.sum(R * torch.sum(torch.square(X.unsqueeze(1) \
                                                              - self.mean.unsqueeze(0)), dim=-1), dim=0) / R.sum(dim=0) #(k, h, w)
        # print(torch.sum(R * torch.sum(torch.square(X.unsqueeze(1) \
        #                                                       - self.mean.unsqueeze(0)), dim=-1), dim=0))
        self.ratio = self.weight / self.deviation #(K, h, w)
        values, idx = torch.sort(self.ratio, dim=0, descending=True)
        cum = torch.cumsum(values, dim=0) #(K, h, w)
        
        # foreground_choice = idx[cum > self.T]
        
    def process(self, frame):
       
        frame_tensor = self.transform(frame).permute(1, 2, 0).unsqueeze(0) #(1, h, w, 3)
        
        dist_square = torch.sum(torch.square(frame_tensor - self.mean), dim=-1) #(k, h, w)
        
        min_dist, index = torch.min(torch.sqrt(dist_square), dim=0) #(h, w)
       
        deviation_choice = torch.gather(self.deviation, dim=0, index = index.unsqueeze(0)) # (h, w)
       
        foreground_mask1 = min_dist < (deviation_choice) * 2.5 # bool (h, w)
        foreground_mask2 = torch.gather(self.foreground_mask, dim=0,index= index.unsqueeze(0))
       
        foreground_mask = torch.logical_and(foreground_mask1, foreground_mask2)
      
        return (foreground_mask2 * 255).numpy().transpose(1, 2, 0)
def check_with_init():
    video_path = './1.mov'
    capture = cv2.VideoCapture(video_path)
    ret, frame = capture.read()
    K = 3
    N = 6
    cnt = 0
    frames = []
    while cnt < N :
        cnt += 1
        ret, frame = capture.read()
        if ret == True:
            cv2.imshow('video', frame) 
            frames.append(frame)
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
        else:
            break
            
    gmm = GMM(frames,n_components=K)
   
  
    capture.release()
    cv2.destroyAllWindows()


def check_with_one_frame():
    video_path = './1.mov'
    capture = cv2.VideoCapture(video_path)
    ret, frame = capture.read()
    K = 3
    if ret == True:
        gmm = GMM(frame, n_components=K)
        w = gmm.w
        h = gmm.h
        gmm.update(frame)
        frame_shown = gmm.process(frame)
        BG = cv2.resize(frame, (w *2, h), interpolation= cv2.INTER_AREA)
        BG[0:h, 0:w] = frame
        BG[0:h, w:2*w] = frame_shown
        cv2.imshow('video',BG)
        
    capture.release()
    cv2.destroyAllWindows()

def check_with_processing():
    video_path = './1.mov'
    capture = cv2.VideoCapture(video_path)
    ret, frame = capture.read()
    K = 3
    N = 3
    cnt = 0
    frames = []
    while cnt < N :
        cnt += 1
        ret, frame = capture.read()
        if ret == True:
            # cv2.imshow('video', frame) 
            frames.append(frame)
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
        else:
            break
            
    gmm = GMM(frames,n_components=K)
    # print(gmm.mean)
    # print(gmm.weight)
    # print(gmm.deviation)
    
    ret, frame = capture.read()
    if ret == True:
        gmm.process(frame)
    capture.release()
    cv2.destroyAllWindows()
# check_with_one_frame()

def main():
    video_path = './video.mp4'
    capture = cv2.VideoCapture(video_path)
    ret, frame = capture.read()
    K = 3
    N = 20
    cnt = 0
    frames = []
    while cnt < N :
        cnt += 1
        ret, frame = capture.read()
        if ret == True:
            # cv2.imshow('video', frame) 
            frames.append(frame)
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
        else:
            break
            
    gmm = GMM(frames,n_components=K)
    
    capture.release()
    capture = cv2.VideoCapture(video_path)
   
    while 1:
        ret, frame = capture.read()
       
        if ret==True:
            frame_shown = gmm.process(frame)
            h = gmm.h
            w = gmm.w
            BG = cv2.resize(frame, (w *2, h), interpolation= cv2.INTER_AREA)
            BG[0:h, 0:w] = frame
            BG[0:h, w:2*w] = frame_shown
            cv2.imshow('video', BG)
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
        else:
            break
            # output.write(BG)
    capture.release()
    cv2.destroyAllWindows()
# check_with_init()
main()

# check_with_processing()


