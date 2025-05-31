import random
from card import Card

class Deck:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cards = [Card(rank, suit) for rank in Card.ranks for suit in Card.suits]
        random.shuffle(self.cards)

    def deal(self, num=1):
        dealt = self.cards[:num]
        self.cards = self.cards[num:]
        return dealt
