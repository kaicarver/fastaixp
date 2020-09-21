# try running motebook software locally and in a regular program
from fastai.vision.all import *

print(URLs.PETS)
path = untar_data(URLs.PETS)/'images'
print(path)
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))
print(dls)
