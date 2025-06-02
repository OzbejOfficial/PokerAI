class Card:
    suits = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    suit_symbols = {
        'Spades': '♠️',
        'Hearts': '♥️',
        'Diamonds': '♦️',
        'Clubs': '♣️'
    }

    def __init__(self, rank, suit):
        if rank not in self.ranks or suit not in self.suits:
            raise ValueError(f"Invalid card: {rank} of {suit}")
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank}{self.suit_symbols[self.suit]}"


    def __str__(self):
        return f"{self.rank} {self.suit}"