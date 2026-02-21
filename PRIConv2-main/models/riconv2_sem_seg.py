import torch
import torch.nn as nn
import torch.nn.functional as F
from riconv2_utils import RIConv2SetAbstraction, RIConv2FeaturePropagation_v2

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 64 if normal_channel else 0
        self.normal_channel = normal_channel

        self.sa0 = RIConv2SetAbstraction(npoint=512, radius=0.2, nsample=8, in_channel=0 + in_channel, mlp=[64], group_all=False)
        self.sa1 = RIConv2SetAbstraction(npoint=256, radius=0.4, nsample=16, in_channel=64 + in_channel, mlp=[128], group_all=False)
        self.sa2 = RIConv2SetAbstraction(npoint=128, radius=0.6, nsample=32, in_channel=128 + in_channel, mlp=[256], group_all=False)
        self.sa3 = RIConv2SetAbstraction(npoint=64, radius=0.8, nsample=64, in_channel=256 + in_channel, mlp=[512], group_all=False)

        self.fp3 = RIConv2FeaturePropagation_v2(radius=1.5, nsample=8, in_channel=512 + 64, in_channel_2=512 + 256, mlp=[512], mlp2=[512])
        self.fp2 = RIConv2FeaturePropagation_v2(radius=0.8, nsample=16, in_channel=512 + 64, in_channel_2=512 + 128, mlp=[512], mlp2=[256])
        self.fp1 = RIConv2FeaturePropagation_v2(radius=0.48, nsample=32, in_channel=256 + 64, in_channel_2=256 + 64, mlp=[256], mlp2=[128])
        self.fp0 = RIConv2FeaturePropagation_v2(radius=0.48, nsample=32, in_channel=128 + 64, in_channel_2=128 + 16, mlp=[128], mlp2=[])

       # Segmentation Head with additional layers
        self.conv1 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.segmentation_head = nn.Conv1d(256, num_class, 1)

        # Reconstruction Head with additional layers
        self.recon_conv1 = nn.Conv1d(128, 256, 1)
        self.recon_bn1 = nn.BatchNorm1d(256)
        self.recon_conv2 = nn.Conv1d(256, 128, 1)
        self.recon_bn2 = nn.BatchNorm1d(128)
        self.recon_conv3 = nn.Conv1d(128, 64, 1)
        self.recon_bn3 = nn.BatchNorm1d(64)
        self.recon_conv4 = nn.Conv1d(64, 3, 1)
        
    def forward(self, xyz):
        B, N, C = xyz.shape
        if self.normal_channel:
            norm = xyz[:, :, 3:6]
            xyz = xyz[:, :, :3]
        else:
            norm = None

        l0_xyz, l0_norm, l0_points = self.sa0(xyz, norm, None)
        l1_xyz, l1_norm, l1_points = self.sa1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sa2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sa3(l2_xyz, l2_norm, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_norm, l3_norm, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_norm, l2_norm, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_norm, l1_norm, l0_points, l1_points)
        l0_points = self.fp0(xyz, l0_xyz, norm, l0_norm, None, l0_points)

        # Segmentation head
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = F.relu(self.bn2(self.conv2(feat)))
        x = self.drop1(feat)
        seg_output = self.segmentation_head(x)
        seg_output = F.log_softmax(seg_output, dim=1)
        seg_output = seg_output.permute(0, 2, 1)

        # Reconstruction head
        recon_feat = F.relu(self.recon_bn1(self.recon_conv1(l0_points)))
        recon_feat = F.relu(self.recon_bn2(self.recon_conv2(recon_feat)))
        recon_feat = F.relu(self.recon_bn3(self.recon_conv3(recon_feat)))
        recon_output = self.recon_conv4(recon_feat)
        recon_output = recon_output.permute(0, 2, 1)

        return seg_output, recon_output, l3_points

"""
class get_loss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(get_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, seg_pred, recon_pred, seg_target, recon_target, weight=None):
        seg_loss = F.nll_loss(seg_pred, seg_target, weight=weight)
        recon_loss = F.mse_loss(recon_pred, recon_target)
        total_loss = self.alpha * seg_loss + self.beta * recon_loss

        return total_loss, seg_loss, recon_loss
"""
class get_loss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(get_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, seg_pred, recon_pred, seg_target, recon_target, weight=None):
        # Calculate NLL loss for segmentation
        seg_loss = F.nll_loss(seg_pred, seg_target, weight=weight, reduction='none')
        
        # Check for NaN in segmentation loss
        if torch.isnan(seg_loss).any():
            print("Segmentation loss contains NaN values")

        # Convert NLL loss back to probability
        seg_prob = torch.exp(-seg_loss)
        epsilon = 1e-6
        seg_prob = torch.clamp(seg_prob, epsilon, 1 - epsilon)
        
        # Check for NaN in probabilities
        if torch.isnan(seg_prob).any():
            print("Segmentation probabilities contain NaN values")

        # Compute focal loss
        focal_loss = ((1 - seg_prob) ** self.gamma) * seg_loss
        focal_loss = focal_loss.mean()

        # Calculate L1 loss for reconstruction
        recon_loss = F.l1_loss(recon_pred, recon_target)

        # Combine the losses
        total_loss = self.alpha * focal_loss + self.beta * recon_loss

        # Check for NaN in total loss
        if torch.isnan(total_loss).any():
            print("Total loss contains NaN values")

        return total_loss, focal_loss, recon_loss

if __name__ == '__main__':
    model = get_model(3, normal_channel=True)
    xyz = torch.rand(6, 1024, 9)  # Example input: batch size 6, 9 features (x, y, z, r, g, b, x_perturbation, y_perturbation, z_perturbation), 1024 points
    seg_output, recon_output, _ = model(xyz)
    print("Segmentation Output Shape: ", seg_output.shape)  # Should be (6, 1024, 3)
    print("Reconstruction Output Shape: ", recon_output.shape)  # Should be (6, 1024, 3)
