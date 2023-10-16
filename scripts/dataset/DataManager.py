import os
import point_cloud_utils as pcu
import json
import trimesh

class LeafScanManager():
    def __init__(self, root_path):
        self.root_path = root_path
        self.neutral_path = os.path.join(root_path, 'canonical/')
        self.species_dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d)) and d !='canonical']

        
    def get_all_species(self):
        species = os.listdir(self.neutral_path)
        name = [os.path.splitext(os.path.basename(specie))[0] for specie in species if '.obj' in specie]
        species = [name.split('_')[0] for name in name]
        return species
    
    def get_poses(self, species):
        species_path = os.path.join(self.root_path, species)
        poses = [pose for pose in os.listdir(species_path) if not '.mtl' in pose and not 'npy'  in pose]
        return sorted(poses)
    
    def get_all_neutral(self):
        neutral_fold = os.path.join(self.root_path, 'canonical')
        nertral_list = [os.path.join(neutral_fold, file) for file in os.listdir(neutral_fold) if '.obj' in file]
        return nertral_list
    
    def get_neutral_pose(self, species):  
        return  f"{species}_canonical.obj"
    
    def get_neutral_path(self):
        return self.neutral_path

    def load_mesh(self,path):
        mesh = pcu.TriangleMesh()
        v, f = pcu.load_mesh_vf(path)
        mesh.vertex_data.positions = v
        mesh.face_data.vertex_ids = f
        return mesh
    
    def get_train_shape_file(self, species):
        file_path = os.path.join(self.neutral_path,'train_file' )
        filename = f"{species}_neutral.npy"
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
            obj_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.obj')]
            for obj_file in obj_files:
                obj_list.append({folder_name: obj_file})

        return obj_list

class LeafImageManger():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.all_species = os.listdir(root_dir)
        with open('dataset/LeafData/train_shape.json', 'r') as f:
            self.train_label = json.load(f)
    def get_all_trainfile_healthy(self):
        
        pass
    
    def get_all_mask(self):
        all_mask = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.JPG') and 'mask' in file:
                    all_mask.append(os.path.join(root,file))
        return all_mask
                    
    def get_mask_train(self):
        healthy_mask_list = []
        disased_mask_list = []
        for species in self.all_species:
            sub_data = self.train_label.get(species,{})
            disased_mask = sub_data.get('diseased', [])
            healthy_mask = sub_data.get('healthy', [])
            healthy_mask_list.extend(healthy_mask)
            disased_mask_list.extend(disased_mask)
        return healthy_mask_list, disased_mask_list

    def get_species_to_idx(self):
        dir_list = os.listdir(self.root_dir)
        species_list = [folder for folder in dir_list if os.path.isdir(os.path.join(self.root_dir, folder))]
        species_to_idx = {}  # 创建一个空字典来保存映射
        species_to_idx = {species: idx for idx, species in enumerate(species_list)}
        return species_to_idx
    
    def get_all_species(self):
        dir_list = os.listdir(self.root_dir)
        species_list = [folder for folder in dir_list if os.path.isdir(os.path.join(self.root_dir, folder))]
        return species_list
    
    def load_mesh(self,path):
        mesh = pcu.TriangleMesh()
        v, f = pcu.load_mesh_vf(path)
        mesh.vertex_data.positions = v
        mesh.face_data.vertex_ids = f
        return mesh

    def get_all_mesh(self):
        all_mesh = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if 'obj' in file and 'healthy' in file:
                    all_mesh.append(os.path.join(root, file))
        
        return all_mesh
    def extract_info_from_meshfile(self, file):
        base, filename = os.path.split(file)
        mesh =self.load_mesh(file)
        _, attribute = os.path.split(base)
        _, species = os.path.split(os.path.dirname(base))
        ret = {
            'mesh': mesh,
            'attribute': attribute,
            'species': species
        }
        return ret
        
    
if __name__ == "__main__":
    manager = LeafScanManager('/home/yang/projects/parametric-leaf/dataset/LeafData')
    # all_species = manager.get_all_species()
    # for species in all_species:
    #     poses = manager.get_poses(species)
    #     neutral = manager.get_neutral_pose(species)
    # pass
    img_root = '/home/yang/projects/parametric-leaf/dataset/LeafData'
    imanager = LeafImageManger(img_root)
    all_mask = imanager.get_all_mask()
    #healthy, diseased = imanager.get_mask_train()
    all_mesh = imanager.get_all_mesh()
    train_dict = imanager.extract_info_from_meshfile(all_mesh[0])
    pass
    