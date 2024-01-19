import torch
import torch.nn as nn
import torch.nn.functional as F

# 1D conv usage:
# batch_size (N) = #3D objects , channels = features, signal_lengt (L) (convolution dimension) = #point samples
# kernel_size = 1 i.e. every convolution over only all features of one point sample


# 3D Single View Reconsturction (for 256**3 input voxelization) --------------------------------------
# ----------------------------------------------------------------------------------------------------

class NDF(nn.Module):


    def __init__(self,hidden_dim=32):
        super(NDF, self).__init__()
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='replicate')  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='replicate')  # out: 128
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate')  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='replicate')  # out: 64
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='replicate')  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='replicate')  # out: 32
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 16
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 8
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 8

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 + 128) * 7 + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()
        self.fc_shape_1 = nn.Conv1d(640, 512, 1)
       # self.fc_shape_2 = nn.Conv1d(1024, 512, 1)
        self.fc_pose_1 = nn.Conv1d(640, 200, 1)
      #  self.fc_pose_2 = nn.Conv1d(1024, 200, 1)
        
        

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def encoder(self,x):
        x = x.unsqueeze(1)
        f_0 = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        net = self.maxpool(net)  # out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        f_5 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        f_6 = net

        return f_0, f_1, f_2, f_3, f_4, f_5, f_6
    
    def encoder_inversion(self,x):
        x = x.unsqueeze(1)
        f_0 = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        net = self.maxpool(net)  # out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        f_5 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        f_6 = net # (5,128,4,4,4)
        
        feature = f_6.view(1,-1).unsqueeze(-1)
        lat_shape = self.fc_shape_1(feature)
       # lat_shape = self.fc_shape_2(lat_shape)
        lat_pose = self.fc_pose_1(feature)
     #   lat_pose = self.fc_pose_2(lat_pose)
        return lat_shape.squeeze(-1), lat_pose.squeeze(-1)

    def decoder(self, p, f_0, f_1, f_2, f_3, f_4, f_5, f_6):

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)

        # feature extraction
        feature_0 = F.grid_sample(f_0, p, padding_mode='replicate')
        feature_1 = F.grid_sample(f_1, p, padding_mode='replicate')
        feature_2 = F.grid_sample(f_2, p, padding_mode='replicate')
        feature_3 = F.grid_sample(f_3, p, padding_mode='replicate')
        feature_4 = F.grid_sample(f_4, p, padding_mode='replicate')
        feature_5 = F.grid_sample(f_5, p, padding_mode='replicate')
        feature_6 = F.grid_sample(f_6, p, padding_mode='replicate')

        # here every channel corresponds to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.actvn(self.fc_out(net))
        out = net.squeeze(1)

        return  out

    def forward(self, p, x):
        out = self.decoder(p, *self.encoder(x))
        return out
    
    
if __name__ == "__main__":
    device = torch.device('cuda')
    model = NDF()
    model.cuda()
    points = torch.randn([5,256,256,256]).to(device)
    latent = torch.randn([512]).to(device)
    model(x=points,p = latent)
    pass