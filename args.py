import argparse

parser = argparse.ArgumentParser(description="DCGAN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
parser.add_argument("--epochs", default=5, type=int, dest="epochs")
parser.add_argument("--device", default="cuda", type=str, dest="device")
parser.add_argument("--channel_n", default="1", type=str, dest="channel_n")
parser.add_argument("--noise_dim", default="100", type=str, dest="noise_dim")

args = parser.parse_args()
