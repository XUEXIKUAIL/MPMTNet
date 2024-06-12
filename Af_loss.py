import torch.nn.functional as F
import torch
def Af(y_s,y):
    # Normalise representations
    #print(sum(y_s[0][0][0]))
    b1, c1, h1, w1 = y_s.shape
    y_s = F.interpolate(y_s, size=(h1, h1), mode='bilinear')
    z_s_norm = F.normalize(y_s, dim=1)
    #print(sum(z_s_norm[0][0][0]))
    # Compute correlation-matrices
    b,c,h,w=z_s_norm.shape
    z_s_norm=z_s_norm.reshape(b,c,h*w).permute(0,2,1)
    z_s_norm_t=z_s_norm.permute(0,2,1)
    c_ss = torch.bmm(z_s_norm, z_s_norm_t)

    y_d = F.interpolate(y, size=(h1, h1), mode='bilinear')
    b, c, h, w = y_d.shape
    y_d = y_d.reshape(b, c, h * w).permute(0, 2, 1)
    y_d_t = y_d.permute(0, 2, 1)
    yy = torch.bmm(y_d, y_d_t)
    loss=0.0
    loss += torch.log2(c_ss.pow(2).sum()) / (h*h*w*w)
    loss -= torch.log2((c_ss * yy).pow(2).sum()) / (h*h*w*w)
    return loss