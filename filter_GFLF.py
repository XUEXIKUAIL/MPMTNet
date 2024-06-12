import cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import math

class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=3, sigma=1.):
        super(GaussianFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2

        self.weight = self._gaussian_kernel()

    def forward(self, x):
        # expand the kernel weights to match the number of input channels
        weight = self.weight.expand(x.size(1), x.size(1), self.kernel_size, self.kernel_size).cuda()

        # apply the convolution with the gaussian kernel
        return F.conv2d(x, weight, padding=self.padding)

    def _gaussian_kernel(self):
        # create a 1D Gaussian kernel
        kernel = torch.exp(-(torch.arange(self.kernel_size) - self.kernel_size // 2) ** 2 / (2 * self.sigma ** 2))

        # normalize the kernel weights
        kernel /= kernel.sum()

        # create a 2D Gaussian kernel by outer product of the 1D kernel
        kernel = torch.outer(kernel, kernel)

        # reshape the kernel to 4D tensor (1, 1, kernel_size, kernel_size)
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)

        # convert the kernel to a PyTorch tensor
        # kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel.clone().detach().requires_grad_(True).to(torch.float32)
        return kernel

class LaplacianFilter(nn.Module):
    def __init__(self, kernel_size=3):
        super(LaplacianFilter, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.weight = self._laplacian_kernel()

    def forward(self, x):
        # expand the kernel weights to match the number of input channels
        weight = self.weight.expand(x.size(1), x.size(1), self.kernel_size, self.kernel_size).cuda()

        # apply the convolution with the laplacian kernel

        return F.conv2d(x, weight, padding=self.padding)

    def _laplacian_kernel(self):
        # create a 2D Laplacian kernel
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)

        # reshape the kernel to 4D tensor (1, 1, kernel_size, kernel_size)
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)

        return kernel
# def gaussian_filter(image, kernel_size, sigma):
#     # 构建高斯核
#     # image = torch.squeeze(image, dim=0)
#     image = image.cpu().detach().numpy()
#     kernel = np.zeros([kernel_size, kernel_size])
#     center = kernel_size // 2
#     for i in range(kernel_size):
#         for j in range(kernel_size):
#             kernel[i, j] = np.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))
#             kernel_sum = np.sum(kernel)
#     kernel = kernel / kernel_sum
#
#     # 卷积操作
#     for i in range(image.shape[0]):
#         print('i', i)
#         filtered_image = cv2.filter2D(image[i], -1, kernel)
#     filtered_image = torch.from_numpy(filtered_image)
#
#     print('filtered_image', filtered_image.shape)
#     # filtered_image = torch.unsqueeze(filtered_image, dim=0)
#     return filtered_image
#
#
# # def padding(img, K_size=3):
# #     # img 为需要处理图像
# #     # K_size 为滤波器也就是卷积核的尺寸，这里我默认设为3*3，基本上都是奇数
# #
# #     # 获取图片尺寸
# #     H, W, C = img.shape
# #
# #     pad = K_size // 2  # 需要在图像边缘填充的0行列数，
# #     # 之所以我要这样设置，是为了处理图像边缘时，滤波器中心与边缘对齐
# #
# #     # 先填充行
# #     rows = np.zeros((pad, W, C), dtype=np.uint8)
# #     # 再填充列
# #     cols = np.zeros((H + 2 * pad, pad, C), dtype=np.uint8)
# #     # 进行拼接
# #     img = np.vstack((rows, img, rows))  # 上下拼接
# #     img = np.hstack((cols, img, cols))  # 左右拼接
# #
# #     return img
# #
# # def laplacian(img, K_size=3):
# #     # 获取图像尺寸
# #     H, W, C = img.shape
# #
# #     # 进行padding
# #     pad = K_size // 2
# #     out = padding(img, K_size=3)
# #
# #     # 滤波器系数
# #     K = np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
# #
# #     # 进行滤波
# #     tem = out.copy()
# #
# #     for h in range(H):
# #         for w in range(W):
# #             for c in range(C):
# #                 out[pad + h, pad + w, c] = np.sum(K * tem[h:h + K_size, w:w + K_size, c], dtype=np.float)
# #
# #     out = np.clip(out, 0, 255)
# #
# #     out = out[pad:pad + H, pad:pad + W].astype(np.uint8)
# #
# #     return out
#
# def laplacian_filter(image):
#     # 使用OpenCV内置的Laplacian函数应用滤波器
#     # filtered_image = cv2.Laplacian(image, cv2.CV_64F)
#     # filtered_image = np.uint8(np.absolute(filtered_image))
#     # image = torch.squeeze(image, dim=0)
#     image = image.detach().numpy()
#     laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
#     for i in image.shape[0]:
#         filtered_image = cv2.filter2D(i, -1, laplacian_kernel)
#
#     filtered_image = torch.from_numpy(filtered_image)
#     # filtered_image = torch.unsqueeze(filtered_image, dim=0)
#
#     return filtered_image


def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter

class LHFA(nn.Module):
    def __init__(self, C, H, W):
        super(LHFA, self).__init__()
        # print('chw', C, H, W)
        self.GSF = GaussianFilter()
        self.LPF = LaplacianFilter()
        self.DCT = MultiSpectralAttentionLayer(C, H, W, reduction=16, freq_sel_method='top16')
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input):
        # print('input', input.shape)
        low_pass = self.GSF(input)
        # print('low_pass', low_pass.shape)
        A_map = self.softmax(self.GAP(self.DCT(low_pass)))
        high_input = low_pass * A_map
        high_pass = self.LPF(high_input)
        out = high_pass + input
        return out

if __name__ == '__main__':

    y_s = torch.randn(2, 128, 120, 160).cuda()

    # img = cv2.cvtColor(np.asarray(y_s), cv2.COLOR_RGB2BGR)
    print('ys', y_s.shape)
    # print('img', img.shape)
    model_filter = LHFA(y_s).cuda()
    out = model_filter(y_s)
    print(out.shape)
    # new_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    # print(new_img.shape)

# import matplotlib.pyplot as plt
# N = 100
# t = np.linspace(0,20,N, endpoint=False)
# x = np.exp(-t/3)*np.cos(2*t)
# y = dct(x, norm='ortho')
#
# window = np.zeros(N)
# window[:20] = 1
# yr = cv2.idct(y*window, norm='ortho')
# sum(abs(x-yr)**2) / sum(abs(x)**2)  # 0.0009872817275276098
# plt.plot(t, x, '-bx')
# plt.plot(t, yr, 'ro')
# window = np.zeros(N)
# window[:15] = 1
# yr = cv2.idct(y*window, norm='ortho')
# sum(abs(x-yr)**2) / sum(abs(x)**2)  #0.06196643004256714
# plt.plot(t, yr, 'g+')
# plt.legend(['x', '$x_{20}$', '$x_{15}$'])
# plt.grid()
# plt.show()

# # 读取图像
# img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
# # 显示结果
# cv2.imshow('Original Image', img)
# cv2.imshow('Laplacian Filtered Image', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()