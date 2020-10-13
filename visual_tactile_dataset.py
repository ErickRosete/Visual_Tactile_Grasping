import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from pathlib import Path
from torch.utils.data import Dataset
import h5py
import torch
import deepdish as dd

class VisualTactileDataset(Dataset):
    def __init__(self, dir, load_data = False, data_cache_size = 5, extension='h5'):
        """
        Visual-Tactile image dataset

        Input params:
            dir: (str) Absolute path of the directory where the data is saved
            load_data: (bool) True -> Loads all data into RAM
                              False -> Lazy loading
            data_cache_size: (int) Number of files cached simultaneously
            extension: (str) Extension of the file names to load 
         """
        if(extension != "h5"):
            raise RuntimeError('Not yet implemented')

        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        
        # List all file names
        dir_path = Path(dir)
        assert(dir_path.is_dir())
        file_names = sorted(dir_path.glob('*.%s' % extension))
        if len(file_names) < 1:
            raise RuntimeError('No %s datasets found' % extension)

        for file_name in file_names:
            self._add_data_infos(str(file_name.resolve()), load_data)

    def group_to_dict(self, group):
        #Save the following fields
        fields = [
            'is_gripping',
            'kinectA_rgb_before',
            'kinectA_rgb_during',
            'gelsightA_before',
            'gelsightA_during',
            'gelsightB_before',
            'gelsightB_during',
            ]
        
        data = {}
        for field in fields:
            data[field] = group[field]

        # For h5py
        # for dname, ds in group.items():
        #     if dname in fields:
        #         # data[dname] = ds[()]
        return data

    def transform(self, img1, img2):
        img1 = transforms.ToPILImage()(img1).convert("RGB") 
        img2 = transforms.ToPILImage()(img2).convert("RGB")

        resize = transforms.Resize(size=(256, 256))
        img1 = resize(img1)
        img2 = resize(img2)

        i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=(224, 224))
        img1 = TF.crop(img1, i, j, h, w)
        img2 = TF.crop(img2, i, j, h, w)

        if random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)

        img1 = transforms.ToTensor()(img1)
        img2 = transforms.ToTensor()(img2)
        return img1, img2

    def _add_data_infos(self, file_path, load_data): 
        if load_data:
            file = dd.io.load(file_path)
            for group in file:
                data = self.group_to_dict(group)
                idx = self._add_to_cache(data, file_path)
                #Adds an element to the list per every element in the dataset
                self.data_info.append({'file_path': file_path, 
                                        'type': "data",
                                        'cache_idx': idx})
        else:
            # Faster with h5py file but unable to read labels
            with h5py.File(file_path, 'r') as h5_file:
                for _, group in h5_file.items(): 
                    for _, _ in group.items():
                        idx = -1  # if data is not loaded its cache index is -1
                        self.data_info.append({'file_path': file_path, 
                                                'type': "data",
                                                'cache_idx': idx})

    def get_data(self, type, i):
        """ Access a chunk of data from the dataset. 
            It makes sure that the data is loaded in case it is not part of the data cache.
        """

        # Check if file_path is in cache
        di = self.get_data_infos(type)[i]
        fp = di['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        idx = di['cache_idx']
        return self.data_cache[fp][idx]
   

    def _load_data(self, file_path):
        """Load data to the cache given the file path and update the cache index in the
        data_info structure.
        """
        file = dd.io.load(file_path)
        for group in file:
            data = self.group_to_dict(group)
            idx = self._add_to_cache(data, file_path)

            # === Update the cache idx ===
            # Find the first index of the data list obtained from file_path
            file_idx = next(i for i, v in enumerate(self.data_info) if v['file_path'] == file_path)
            self.data_info[file_idx + idx]['cache_idx'] = idx

        # Remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path) # Do not delete file needed
            self.data_cache.pop(removal_keys[0])
            # Change cache_idx to -1 in data info of entries with removed key
            for di in self.data_info:
                if di['file_path'] == removal_keys[0]:
                    di['cache_idx'] = -1

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. 
        There is one cache list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1


    def __getitem__(self, index):
        # Get data
        cached_data = self.get_data("data", index)
        
        # Transforms 
        cam1, cam2 = self.transform(cached_data['kinectA_rgb_before'], cached_data['kinectA_rgb_during'])
        lg1, lg2 = self.transform(cached_data['gelsightA_before'], cached_data['gelsightA_during'])     
        rg1, rg2 = self.transform(cached_data['gelsightB_before'], cached_data['gelsightB_during'])         
        lg1 = torch.abs(lg2 - lg1)
        rg1 = torch.abs(rg2 - rg1)
    
        data = {}
        data['cam_bef'] = cam1
        data['cam_dur'] = cam2
        data['lgel_bef'] = lg1
        data['lgel_dur'] = lg2
        data['rgel_bef'] = rg1
        data['rgel_dur'] = rg2

        label = int(cached_data['is_gripping'])

        return data, label


    def get_data_infos(self, type):
        """ Get data infos belonging to a certain type of data."""
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type   

    def __len__(self):
        return len(self.get_data_infos('data'))
