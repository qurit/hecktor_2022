
class SegmentationDatasetPTCT(Dataset):
    def __init__(self, inputs_pt, inputs_ct, targets):
        self.inputs_pt = inputs_pt 
        self.inputs_ct = inputs_ct 
        self.targets = targets
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.int64
        
        # self.normalize = torchvision.transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],               
        #     std=[0.229, 0.224, 0.225])

    
    def __len__(self):
        return len(self.inputs_pt)
    
    def __getitem__(self, index):
        
        ctimg_path = self.inputs_ct[index]
        ptimg_path = self.inputs_pt[index]
        gtimg_path = self.targets[index]
        
        xct, _ = utils.nib2numpy(ctimg_path) # get image 2D np array
        xpt, _ = utils.nib2numpy(ptimg_path) 
        y, _ = utils.nib2numpy(gtimg_path)

        '''
        Transformations:
        - Resize
        - trunc_normalize (Ivan/Yixi preprocess)
        - Stack 3 slices to make 3 channel image (first two channels PT, 3rd channel CT)
        - permute axes to get first axes as channel
        '''
       
        xct = resize_2dtensor_bilinear(xct)
        xpt = resize_2dtensor_bilinear(xpt)
        y = resize_2dtensor_nearest(y)
      
        xpt = trunc_scale_preprocess_pt(xpt)
        xct = trunc_scale_preprocess_ct(xct) 

       
        x = np.dstack((xpt, xpt, xct))

        # make first axes as number of channels
        x = np.moveaxis(x, source=-1, destination=0)
        
        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
      
        return x, y