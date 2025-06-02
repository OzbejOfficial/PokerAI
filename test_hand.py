from hand import Hand
from card import Card

hand1 = [Card('A', 'Spades'), Card('K', 'Hearts')]
hand2 = [Card('10', 'Diamonds'), Card('K', 'Clubs')]

community_cards = []

h1 = Hand(hand1, community_cards)
h2 = Hand(hand2, community_cards)

print("Hand 1:", h1)
print("Hand 2:", h2)
print("Comparison result:", h1.compare(h2))
print("Hand 1 score:", h1.score)
print("Hand 2 score:", h2.score)