import argparse
import logging
import os
import torch
from PIL import Image

from unet.unet_model import UNet
from utils.data_vis import plot_weights


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='checkpoints\CP_epoch5.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    # parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
    #                     help='Filenames of ouput images')
    # parser.add_argument('--viz', '-v', action='store_true',
    #                     help="Visualize the images as they are processed",
    #                     default=False)
    # parser.add_argument('--no-save', '-n', action='store_true',
    #                     help="Do not save the output masks",
    #                     default=False)
    # parser.add_argument('--mask-threshold', '-t', type=float,
    #                     help="Minimum probability value to consider a mask pixel white",
    #                     default=0.5)
    # parser.add_argument('--scale', '-s', type=float,
    #                     help="Scale factor for the input images",
    #                     default=0.5)
    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


if __name__ == '__main__':
    args = get_args()
    # out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    # Visualize layer
    plot_weights(net, 'conv12')
