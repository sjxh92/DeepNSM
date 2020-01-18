import argparse

parser = argparse.ArgumentParser(
    description='DeepNSM'
)

parser.add_argument('--node-capacity', type=int, default=100,
                    help="节点的容量")
parser.add_argument('--link-capacity', type=int, default=100,
                    help="链接的容量")
parser.add_argument('--node-number', type=int, default=6,
                    help="节点的数量")
parser.add_argument('--link-number', type=int, default=20,
                    help="链接的数量")
parser.add_argument('--window', type=int, default=10,
                    help="窗口")

# -------------------------------------------------------------
parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
parser.add_argument('--env', type=str, default='NSMGame')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed [0, 2 ** 32)')
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--final-exploration-steps',
                    type=int, default=10 ** 4)
parser.add_argument('--start-epsilon', type=float, default=1.0)
parser.add_argument('--end-epsilon', type=float, default=0.1)
parser.add_argument('--noisy-net-sigma', type=float, default=None)
parser.add_argument('--demo', action='store_true', default=False)
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--steps', type=int, default=10 ** 2)
parser.add_argument('--prioritized-replay', action='store_true')
parser.add_argument('--replay-start-size', type=int, default=1000)
parser.add_argument('--target-update-interval', type=int, default=10 ** 2)
parser.add_argument('--target-update-method', type=str, default='hard')
parser.add_argument('--soft-update-tau', type=float, default=1e-2)
parser.add_argument('--update-interval', type=int, default=1)
parser.add_argument('--eval-n-runs', type=int, default=100)
parser.add_argument('--eval-interval', type=int, default=10 ** 4)
parser.add_argument('--n-hidden-channels', type=int, default=100)
parser.add_argument('--n-hidden-layers', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--minibatch-size', type=int, default=None)
# parser.add_argument('--render-train', action='store_true')
# parser.add_argument('--render-eval', action='store_true')
parser.add_argument('--monitor', action='store_true')
parser.add_argument('--reward-scale-factor', type=float, default=1e-3)
args = parser.parse_args()
