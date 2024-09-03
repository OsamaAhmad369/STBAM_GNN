import argparse


def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default='cuda', help="Device to run the model")
    parser.add_argument("--nodes", type=int, default=64, help="Number of nodes for superpixels")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bs", type=int, default=3)
    parser.add_argument("--penalty", type=float, default=1e-4)
    parser.add_argument("--lrate", type=float, default=1e-4)
    parser.add_argument("--mode",type=str,default="train", help="For evaluation set mode test otherwise train")
    parser.add_argument("--weightspath", type=str, default='')
    return parser
