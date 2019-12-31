import argparse

parser = argparse.ArgumentParser(
    description='DeepNSM'
)

parser.add_argument('--node-capacity', type=int, default=100,
                    help="节点的容量")
parser.add_argument('--link-capacity', type=int, default=100,
                    help="链接的容量")
parser.add_argument('--node-number', type=int,default=6,
                    help="节点的数量")
args = parser.parse_args()