import Heuristic as h
import MetroNetwork
import NSMGame


if __name__ == "__main__":
    game = NSMGame.Game(mode="LINN",
                        total_time=150,
                        wave_num=10,
                        vm_num=10,
                        max_iter=40,
                        rou=1,
                        mu=70,
                        k=3,
                        w=3,
                        n=2,
                        weight=1)
    print('--------------------the service requests---------------')
    game.reset()
    game.show_results()
    print('--------------------the network state-------------------')
    game.ffksp()
    #game.show_results()
