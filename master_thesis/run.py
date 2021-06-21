"""
Author: Basel Alyafi
Master Thesis Project
Erasmus Mundus Joint Master in Medical Imaging and Applications

21062021 minor adjustments by Richard Osuala (BCN-AIM)
"""

import torch
import numpy as np
import os
from skimage import io
from DCGAN import Generator


def generate():
    """
        Generate n synthetic mammography images using pretrained weights.
    """

    # the name of the generator (for calcifications we can use 'mass_calcification_gen')
    model_path = 'malign_mass_gen'

    # the path where to save_model the images
    imgs_path = 'generated_images/'

    # number of images to be generated
    n_imgs = 10

    # create an instance of the generator
    model = Generator(ngpu=1, nz=200, ngf=45, nc=1)
    device_as_string = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the pretrained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device_as_string)))

    # run the trained generator to generate n_imgs images at imgs_path
    run_generator(model=model, batch_size=n_imgs, save_path=imgs_path, RGB=False)


def run_generator(model, batch_size, save_path, RGB):  # DOC OK
    """
    to run a generator to generate images and save them.

    Params
    ------
    model: nn.Module
        the model to run
    batch_size: int
        number of images to generate
    save_path: string
        where to save the generated images
    RGB: bool
        if True images will be saved in RGB format, otherwise grayscale will be used

    Returns
    -------
    void
    """
    # detect if there is a GPU, otherwise use cpu instead
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model input
    fixed_noise = torch.randn(batch_size, model.nz, 1, 1, device=device)
    model.to(device)

    # Testing mode
    mode = model.training
    model.eval()
    model.apply(apply_dropout)

    with torch.no_grad():
        output = model(fixed_noise).detach().cpu()

    # back to original training state
    model.train(mode=mode)

    # create the path if does not exist.
    if not (os.path.exists(save_path)):
        os.makedirs(save_path, exist_ok=True)

    print('Running Generator...')
    for i in range(batch_size):

        # rescale intensities from [-1,1] to [0,1]
        img = np.transpose(output[i], [1, 2, 0]) / 2 + 0.5
        # img = np.squeeze(img)
        img = np.array(255 * img).round().astype(np.uint8)

        # if one channel, squeeze to 2d
        if img.shape[2] == 1:
            img = img.squeeze(axis=2)

        # if gray but RGB required
        if RGB and len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        # save the image
        io.imsave(save_path + '{}.png'.format(i), img)
    print('Finished Generating Images')


def apply_dropout(layer):  # DOC OK
    """
    This function is used to activate dropout layers during training

    Params:
    -------
    layer: torch.nn.Module
        the layer for which the dropout to be activated

    Returns:
    --------
    void
    """
    classname = layer.__class__.__name__
    if classname.find('Dropout') != -1:
        layer.train()

if __name__ == "__main__":
    generate()
