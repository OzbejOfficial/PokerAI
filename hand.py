from deuces import Card as DCard, Evaluator
from collections import Counter

suit_map = {'Spades': 's', 'Hearts': 'h', 'Diamonds': 'd', 'Clubs': 'c'}
rank_map = {'10': 'T', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A',
            '2': '2', '3': '3', '4': '4', '5': '5', '6': '6',
            '7': '7', '8': '8', '9': '9'}

class Hand:
    evaluator = Evaluator()

    def __init__(self, hole_cards, community_cards):
        self.hole_cards = hole_cards
        self.community_cards = community_cards
        self.player_cards = [self._to_deuce(c) for c in hole_cards]
        self.board_cards = [self._to_deuce(c) for c in community_cards]
        self.score = self.evaluator.evaluate(self.board_cards, self.player_cards)
        self.rank = self.evaluator.class_to_string(self.evaluator.get_rank_class(self.score))

    def _to_deuce(self, card):
        rank = rank_map.get(card.rank, card.rank)
        suit = suit_map[card.suit]
        return DCard.new(f"{rank}{suit}")

    def compare(self, other):
        return (self.score < other.score) - (self.score > other.score)
    
    def __lt__(self, other):
        return self.score > other.score

    def __eq__(self, other):
        return self.score == other.score


    def __str__(self):
        from deuces import Card as DCard
        hole_str = " ".join([DCard.int_to_pretty_str(c) for c in self.player_cards])
        board_str = " ".join([DCard.int_to_pretty_str(c) for c in self.board_cards])
        return f"{self.rank} | Hole: {hole_str} | Board: {board_str}"

    __repr__ = __str__
