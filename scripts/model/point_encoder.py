import torch
import torch.nn as nn
import torch.nn.functional as F


class PCAutoEncoder(nn.Module):
    """ Autoencoder for Point Cloud 
    Input: 

    Output: 
    """
    def __init__(self, point_dim=3):
        super(PCAutoEncoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=point_dim, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=1)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=200, kernel_size=1)

        # self.fc1 = nn.Linear(in_features=1024, out_features=1024)
        # self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        # self.fc3 = nn.Linear(in_features=512, out_features=num_points*3)
        # self.fc4 = nn.Linear(in_features=200, out_features=num_points*3)

        #batch norm
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(200)
    
    def forward(self, x):

        batch_size = x.shape[0]
        point_dim = x.shape[1]
        num_points = x.shape[2]

        #encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        latent_shape = F.relu(self.bn3(self.conv5(x)))
        latent_deform = F.relu(self.bn4(self.conv6(x)))

        # do max pooling 
        latent_shape = torch.max(latent_shape, 2, keepdim=True)[0]
        latent_shape = latent_shape.view(-1, 512)

        latent_deform = torch.max(latent_deform, 2, keepdim=True)[0]
        latent_deform = latent_deform.view(-1, 200)
        


        return latent_shape, latent_deform




class Imgencoder(nn.Module):
    def __init__(self, dino_model, new_layer_out_features):
        super(Imgencoder, self).__init__()
        self.dino_model = dino_model
        # Remove the head of the DINO model if you only want the features
        self.dino_model.head = nn.Identity()
        
        # Add your new layer
        # Determine the correct input features by inspecting the original model's output features
        self.new_layer = nn.Linear(in_features=384, out_features=512)

        # Optionally, initialize the new layer's weights
        nn.init.xavier_uniform_(self.new_layer.weight)
        nn.init.zeros_(self.new_layer.bias)

    def forward(self, x):
        # Get the features from the pre-trained DINO model
        features = self.dino_model(x)
        
        # Apply the new layer to these features
        output = self.new_layer(features)
        return output

# Instantiate the custom model
# Here, you need to specify the number of output features for your new layer
new_layer_out_features = 512
#custom_model = CustomDinoModel(encoder_2d, new_layer_out_features)

# Now you can use custom_model as your new model with an additional layer

# write a encoder with three conv layer and one fc layer for mask [128,128], output is [7]

# class MaskEncoder(nn.module):
    


if __name__ == "__main__":
    # test the autoencoder
    model = PCAutoEncoder(point_dim=3)
    print(model)
    x = torch.rand((1, 3, 3500))
    latent_shape, latent_deform = model(x)
    pass