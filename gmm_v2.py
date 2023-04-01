import cv2
import torch
class GMM(torch.nn.Module):
    def __init__(self, frame, n_components=3, alpha = 0.2, T = 0.7):
        super(GMM, self).__init__()
        frame_tensor = torch.from_numpy(frame) # (h, w, 3)
        self.h = frame_tensor.shape[0]
        self.w = frame_tensor.shape[1]
        self.mean = torch.zeros(n_components, self.h, self.w, 3)
        self.deviation = torch.ones(n_components, self.h, self.w)
        self.weight = torch.ones(n_components, self.h, self.w) / n_components
        self.alpha = alpha
        self.n_components = n_components
        self.weight_update_mask = torch.zeros(n_components, self.h, self.w)
        self.mean_update_mask = torch.zeros(n_components, self.h, self.w, 3)
        self._prob = torch.zeros(n_components, self.h, self.w)
        self.distanse_mask = torch.zeros(n_components, self.h, self.w)
        self.mask_some = torch.zeros(self.h, self.w)
        self.mask_divide = torch.zeros(n_components,self.h, self.w)
        self.T = T
    
    def to_tensor(self, frame):
        return torch.from_numpy(frame)
    
    def relative_dist(self, frame):
        # print(torch.sum(torch.square(self.to_tensor(frame).unsqueeze(0) - self.mean), dim=-1).shape)
        # print(self.mean.shape)
        # print(self.deviation.shape)
        return torch.sum(torch.square(self.to_tensor(frame).unsqueeze(0) - self.mean), dim=-1) / torch.square(self.deviation) #(K, h, w)
    
    def prob(self, frame):
        temp = self.relative_dist(frame)
        return torch.exp(- temp / 2) / (torch.sqrt(torch.tensor((2 * torch.pi) ** 3)) * self.deviation)
    def update_prob(self, frame):
        self._prob = self.prob(frame)
    def get_dist_mask(self, frame):
        ## mahalanobis_probability
        temp = self.relative_dist(frame)
        distance_mask = torch.sqrt(temp) < 2.5 * self.deviation
        return distance_mask
    
    def update_mask(self, frame):
        self.distance_mask = self.get_dist_mask(frame)
        self.mask_some = self.distance_mask.any(dim=0)
        self.weight_update_mask = torch.where(self.mask_some, self.distance_mask, -1)
        self.mean_update_mask = self.weight_update_mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        ratio = self.weight / self.deviation
        values, idx = ratio.sort(dim=0, descending=True)
        cum = torch.cumsum(values, dim=0)
        # mask_divide = cum < self.T
        mask_divide = (ratio == ratio.max(dim=0))
        # self.mask_divide = torch.gather(mask_divide, dim=0, index= idx)
        self.mask_divide = mask_divide



    
    def update_weight(self):
        weight_update_mask = self.weight_update_mask
        self.weight = torch.where(weight_update_mask == 1, (1 - self.alpha) * self.weight + self.alpha, self.weight)
        self.weight = torch.where(weight_update_mask == 0, (1 - self.alpha) * self.weight, self.weight)

    def update_mean(self, frame):
        X = self.to_tensor(frame).unsqueeze(0).repeat(self.n_components, 1, 1, 1)
        rho = self.alpha * self._prob
        R = rho.unsqueeze(-1).repeat(1, 1, 1, 3)
        self.mean = torch.where(self.mean_update_mask == 1, (1 - R) * self.mean + R * X, self.mean)

        ### Great coding , solve the problem of bias models
        self.mean = torch.where(self.mean_update_mask == -1, X, self.mean)

    def update_deviation(self, frame):
        rho = self.alpha * self._prob
        # print(rho.shape)
        # print((torch.sqrt(1 - rho)* torch.square(self.deviation)).shape)
        # print(self.update_mask.shape)
        self.deviation = torch.where(self.mask_divide, torch.sqrt(1 - rho)* torch.square(self.deviation) + \
                                     rho * torch.sum(torch.square(self.to_tensor(frame) - self.mean), dim=-1), self.deviation)

    def update(self, frame):
        self.update_mask(frame)
        self.update_prob(frame)
        self.update_weight()
        self.update_mean(frame)
        self.update_deviation(frame)

    
    def process(self, frame):
        self.update(frame)
        frame_show = torch.logical_and(self.mask_divide, self.distance_mask)
        frame_show = frame_show.any(dim=0).unsqueeze(-1).repeat(1, 1, 3).numpy() * 255

        return frame_show




        

