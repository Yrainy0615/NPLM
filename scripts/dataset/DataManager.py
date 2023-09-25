import os
import point_cloud_utils as pcu

class LeafScanManager():
    def __init__(self, root_path):
        self.root_path = root_path
        self.neutral_path = os.path.join(root_path, 'canonical/')
        self.species_dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d)) and d !='canonical']
        
    def get_all_species(self):
        return self.species_dirs
    
    def get_poses(self, species):
        species_path = os.path.join(self.root_path, species)
        poses = [pose for pose in os.listdir(species_path) if 'rigid' in pose]
        return sorted(poses)
    
    def get_neutral_pose(self, species):  
        return  f"{species}_canonical_rigid"
    
    def get_neutral_path(self):
        return self.neutral_path

    def load_mesh(self,path):
        mesh = pcu.TriangleMesh()
        v, f = pcu.load_mesh_vf(path)
        mesh.vertex_data.positions = v
        mesh.face_data.vertex_ids = f
        return mesh
    
    def get_train_shape_file(self, species):
        file_path = os.path.join(self.neutral_path,'train' )
        filename = f"{species}_canonical_rigid_neutral.npy"
        train_file = os.path.join(file_path,filename)
        return train_file
    
    def  get_train_pose_file(self, species, idx):
        file_path = self.root_path +'/' + species + '/train'
        name = os.path.splitext(os.path.basename(idx))[0] + '.npy'
        train_file = os.path.join(file_path, name)
        return train_file
    
    def mesh_from_npy_file(self,npy_path):
            # Split the path to remove the filename and get all directories
        parts = os.path.normpath(npy_path).split(os.sep)
        
        # Remove the 'train' directory if it exists in the path
        if 'train' in parts:
            parts.remove('train')
        
        # Join the parts back together to form the path without 'train'
        path_without_train = os.path.join(*parts)
        
        # Replace the .npy extension with .obj
        obj_path = os.path.splitext(path_without_train)[0] + '.obj'
        
        return '/'+obj_path

    
    def get_species_path(self, species):
        return os.path.join(self.root_path, species)

    def get_all_pose(self):
        obj_list = []
        species_dirs = [os.path.join(self.root_path, species) for species in self.species_dirs]
        for folder_path in species_dirs:
            folder_name = os.path.basename(folder_path)
            obj_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.obj') and 'rigid' in filename]
            for obj_file in obj_files:
                obj_list.append({folder_name: obj_file})

        return obj_list

class LeafImageManger():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.all_species = os.listdir(root_dir)
        
    def get_all_trainfile_healthy(self):
        
        pass
    
    def get_all_mask(self):
        all_mask = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.JPG') and 'mask' in file:
                    all_mask.append(file)
        return all_mask
                    

if __name__ == "__main__":
    manager = LeafScanManager('/home/yang/projects/parametric-leaf/dataset/leaf')
    # all_species = manager.get_all_species()
    # for species in all_species:
    #     poses = manager.get_poses(species)
    #     neutral = manager.get_neutral_pose(species)
    # pass
    img_root = '/home/yang/projects/parametric-leaf/dataset/LeafData'
    imanager = LeafImageManger(img_root)
    all_mask = imanager.get_all_mask()
    