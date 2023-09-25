import torch
import torch.nn as nn
import torch.optim as optim
from NPM.scripts.dataset.keypoints_dataset import LeafKeypointDataset
from torch.utils.data import DataLoader
from model.dgcnn import DGCNN_cls 
import argparse
import open3d as o3d
from matplotlib import pyplot as plt
import trimesh

def save_checkpoint(epoch, model, optimizer, filename="checkpoint_no_trans.pth"):
    """Save checkpoint if a new best is achieved"""
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)
    print("=> saved checkpoint '{}' (epoch {})".format(filename, epoch))
    
def train(dataloader, criterion, optimizer,model):
    # Training loop
    model.train()
    num_epochs = 2000
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            outputs,sample = model(data)
            
            # Compute the loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print log
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    save_checkpoint(epoch, model,optimizer)  

def visualize_keypoints(vertex,keypoints):
                fig = plt.figure()
                ax = fig.add_subplot(111,projection='3d')
                ax.set_xlim([-0.5, 0.5])
                ax.set_ylim([-0.5, 0.5])
                ax.set_zlim([-0.5, 0.5])

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                ax.scatter(vertex[:,0], vertex[:,1], vertex[:,2], c= 'r')
                ax.scatter(keypoints[:,0], keypoints[:,1], keypoints[:,2], c= 'b')
                plt.show()
    
def test(model, dataloader, visualize=True):
       model.eval()
       for i, (data, _) in enumerate(dataloader):
            data = data.to(device)
            with torch.no_grad():
                output,sample = model(data)
            # labels = (output>0.8).cpu().numpy().squeeze()
            # keypoints = data[labels == 1]
            vertex = data.detach().cpu().numpy().squeeze(0)
            keypoints = output.detach().cpu().numpy().squeeze(0)
            # visualize 
            if visualize:
                visualize_keypoints(vertex, keypoints)
            
            
           
     
if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--mode', type=str, default='test', metavar='N',
                        choices=['train', 'test', 'inference'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #torch.manual_seed(args.seed)
    # Initialize the dataset and dataloader
    root_path = 'dataset/keypoints_detection'
    
    # Initialize the DGCNN model
    model = DGCNN_cls(args).to(device)
    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    #criterion = nn.CrossEntropyLoss()  # Or whichever loss function you need
    criterion = nn.MSELoss()
    if args.mode == 'train':
        dataset = LeafKeypointDataset(root_path=root_path,transform=False)
        dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)
        train(model=model, dataloader=dataloader,optimizer=optimizer, criterion=criterion)
    if args.mode == 'test':
        dataset = LeafKeypointDataset(root_path=root_path,transform=False)
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=2)
        checkpoint  = torch.load('checkpoint_no_trans.pth')
        model.load_state_dict(checkpoint['state_dict'])
        test(model, dataloader)
        
    if args.mode == 'inference':
        checkpoint  = torch.load('checkpoint_no_trans.pth')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        test_data = 'dataset/id1/id1_20_rigid.obj'
        mesh = trimesh.load_mesh(test_data)
        verts = mesh.vertices
        pred,sample = model(torch.tensor(verts).to(device).unsqueeze(0))
        keypoints = pred.detach().cpu().numpy().squeeze(0)
        sample = sample.detach().cpu().numpy().squeeze(0)
        visualize =True
        if visualize:
            visualize_keypoints(sample, keypoints)
        
        

    

    
    
    
