from game import Game
from player import Human, RandomBot

def main():
    game = Game(verbose=True)

    me = Human("You")
    bot = RandomBot("RandBot")

    game.add_player(me)
    game.add_player(bot)

    while True:
        game.play_round()
        again = input("Play another round? (y/n): ").lower()
        if again != 'y':
            break

if __name__ == "__main__":
    main()
