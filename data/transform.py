import torchvision.transforms as T

def build_transforms(cfg, is_train=True):
    normalize_tf = T.Normalize(mean=cfg.Input_Pixel_MEAN, std=cfg.Input_Pixel_STD)
    transform = T.Compose([T.Resize((cfg.Input_Size[0], cfg.Input_Size[1])), T.ToTensor()])
    return transform