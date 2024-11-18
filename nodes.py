from transformers import AutoModelForImageSegmentation, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
import os
from scipy.ndimage import binary_erosion
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from folder_paths import models_dir, add_model_folder_path, get_folder_paths

torch.set_float32_matmul_precision(["high", "highest"][0])

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

current_path  = os.getcwd()

## ComfyUI portable standalone build for Windows 

add_model_folder_path('birefnet', os.path.join(models_dir, "birefnet"))
model_path = "BiRefNet"

max_gpu_size = 3

def clear_memory():
    import gc
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


def alpha_matting_cutout(
    img: Image,
    mask: Image,
    foreground_threshold: int,
    background_threshold: int,
    erode_structure_size: int,
) -> Image:
    """
    Perform alpha matting on an image using a given mask and threshold values.

    This function takes a PIL image `img` and a PIL image `mask` as input, along with
    the `foreground_threshold` and `background_threshold` values used to determine
    foreground and background pixels. The `erode_structure_size` parameter specifies
    the size of the erosion structure to be applied to the mask.

    The function returns a PIL image representing the cutout of the foreground object
    from the original image.
    """
    if img.mode == "RGBA" or img.mode == "CMYK":
        img = img.convert("RGB")

    img_array = np.asarray(img)
    mask_array = np.asarray(mask)

    is_foreground = mask_array > foreground_threshold
    is_background = mask_array < background_threshold

    structure = None
    if erode_structure_size > 0:
        structure = np.ones(
            (erode_structure_size, erode_structure_size), dtype=np.uint8
        )

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = np.full(mask_array.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    img_normalized = img_array / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)

    return cutout


def naive_cutout(img: Image, mask: Image) -> Image:
    """
    Perform a simple cutout operation on an image using a mask.

    This function takes a PIL image `img` and a PIL image `mask` as input.
    It uses the mask to create a new image where the pixels from `img` are
    cut out based on the mask.

    The function returns a PIL image representing the cutout of the original
    image using the mask.
    """
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask)
    return cutout

def putalpha_cutout(img: Image, mask: Image) -> Image:
    """
    Apply the specified mask to the image as an alpha cutout.

    Args:
        img (PILImage): The image to be modified.
        mask (PILImage): The mask to be applied.

    Returns:
        PILImage: The modified image with the alpha cutout applied.
    """
    img.putalpha(mask)
    return img

colors = ["transparency", "green", "white", "red", "yellow", "blue", "black", "pink", "purple", "brown", "violet", "wheat", "whitesmoke", "yellowgreen", "turquoise", "tomato", "thistle", "teal", "tan", "steelblue", "springgreen", "snow", "slategrey", "slateblue", "skyblue", "orange"]

def get_device_by_name(device):
    """
    "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto"}), 
    """
    if device == 'auto':
        try:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            elif torch.xpu.is_available():
                device = "xpu"
        except:
                raise AttributeError("What's your device(到底用什么设备跑的)？")
    print("\033[93mUse Device(使用设备):", device, "\033[0m")
    return device

_birefnet_model = None

class BiRefNet_Lite:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        
        return {
            "required": {
                "image": ("IMAGE",),
                "load_local_model": ("BOOLEAN", {"default": True}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto"}),
                "cutout_func": (["putalpha", "naive", "alpha_matting"],{"default": "putalpha"}),
                "cached": ("BOOLEAN", {"default": True}),
                "cpu_size": ("FLOAT",{"default": 0}),
                "alpha_matting_foreground_threshold": ("INT", {"default": 240}),
                "alpha_matting_background_threshold": ("INT", {"default": 10}),
                "alpha_matting_erode_size": ("INT", {"default": 10}),
            },
            "optional": {
                "local_model_path": ("STRING", {"default": model_path}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "background_remove"
    CATEGORY = "🔥BiRefNet"
  
    def background_remove(self, 
                          image, 
                          load_local_model,
                          device,
                          cutout_func,
                          cpu_size,
                          cached,
                          *args, **kwargs
                          ):
        processed_images = []
        processed_masks = []
       
        device = get_device_by_name(device)
        birefnet = None
        global _birefnet_model
        if cached and _birefnet_model is not None:
            birefnet = _birefnet_model
        else:
            if _birefnet_model is not None:
                _birefnet_model = None
                clear_memory()
            
            if load_local_model:
                local_model_path = kwargs.get("local_model_path", model_path)
                local_model_path = os.path.join(get_folder_paths('birefnet')[-1], local_model_path)
                # 判断是否开启双卡支持
                # 获取不同显卡的内存可用量
                # 显卡不足时，自动往CPU调度
                spare_params = {}
                config, kwargs = AutoConfig.from_pretrained(
                    local_model_path,
                    return_unused_kwargs=True,
                    trust_remote_code=True,
                    code_revision=None,
                    _commit_hash=None,
                    **kwargs,
                )
                
                class_ref = config.auto_map["AutoModelForImageSegmentation"]
                model_class = get_class_from_dynamic_module(
                    class_ref, local_model_path, code_revision=None, **kwargs
                )
                if cpu_size > 0 and device == 'cuda':
                    spare_params['device_map'] = 'auto'
                    spare_params['max_memory'] = {0: f"{max_gpu_size}GiB", "cpu": f"{cpu_size}GiB"}
                    setattr(model_class, '_no_split_modules', ["Decoder", "SwinTransformer"])
                else:
                    try:
                        if hasattr(model_class, "_no_split_modules"):
                            delattr(model_class, "_no_split_modules")
                    except Exception as e:
                        print('No need to delete:', e)
                    
                birefnet = AutoModelForImageSegmentation.from_pretrained(local_model_path,trust_remote_code=True, **spare_params)
            else:
                birefnet = AutoModelForImageSegmentation.from_pretrained(
                    "ZhengPeng7/BiRefNet", trust_remote_code=True
                )
                # 远程加载不启动分块
                cpu_size = 0
            if cpu_size == 0:
                birefnet.to(device)
        
        for image in image:
            orig_image = tensor2pil(image)
            w,h = orig_image.size
            image = resize_image(orig_image)
            im_tensor = transform_image(image).unsqueeze(0)
            im_tensor=im_tensor.to(device)
            with torch.no_grad():
                result = birefnet(im_tensor)[-1].sigmoid().cpu()
            result = torch.squeeze(F.interpolate(result, size=(h,w)))
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)    
            im_array = (result*255).cpu().data.numpy().astype(np.uint8)
            pil_im = Image.fromarray(np.squeeze(im_array))
            if cutout_func == 'putalpha':
                new_im = putalpha_cutout(orig_image, pil_im)
            elif cutout_func == 'naive':
                new_im = naive_cutout(orig_image, pil_im)
            elif cutout_func == 'alpha_matting':
                new_im = alpha_matting_cutout(
                    orig_image,
                    pil_im,
                    foreground_threshold=kwargs.get('alpha_matting_foreground_threshold'),
                    background_threshold=kwargs.get('alpha_matting_background_threshold'),
                    erode_structure_size=kwargs.get('alpha_matting_erode_size')
                )
            new_im_tensor = pil2tensor(new_im)
            pil_im_tensor = pil2tensor(pil_im)
            
            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)
        
        clear_memory()

        new_ims = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)

        return new_ims, new_masks


NODE_CLASS_MAPPINGS = {
    "BiRefNet_Lite": BiRefNet_Lite
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "BiRefNet_Lite": "🔥BiRefNet_Lite"
}
