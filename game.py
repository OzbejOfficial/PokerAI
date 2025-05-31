from deck import Deck
from hand import Hand

from colorama import Fore, Style, init
init(autoreset=True)

class Game:
    def __init__(self, verbose=False):
        self.players = []
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.verbose = verbose
        self.starting_bet = 10
        self.round_over = False

    def add_player(self, player):
        self.players.append(player)

    def reset_for_new_round(self):
        self.deck.reset()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.round_over = False
        for player in self.players:
            if player.stack <= 0:
                player.stack = 1000
            player.reset_for_round()
            if player.stack >= self.starting_bet:
                player.stack -= self.starting_bet
                player.bet = self.starting_bet
                self.pot += self.starting_bet
            else:
                player.folded = True

    def deal_hole_cards(self):
        for player in self.players:
            player.hand = self.deck.deal(2)

    def deal_community_cards(self, n):
        self.community_cards += self.deck.deal(n)

    def betting_round(self):
        print(Fore.MAGENTA + "\n--- Betting Round ---")
        print(Fore.MAGENTA + f"Community Cards: {self.community_cards}")
        if self.verbose:
            for p in self.players:
                print(f"{p.name} stack: {p.stack}, folded: {p.folded}, bet: {p.bet}")

        # Skip if round is already over
        if self.round_over:
            return

        while True:
            active = [p for p in self.players if not p.folded]
            not_allin = [p for p in active if p.stack > 0]
            if len(active) == 1:
                winner = active[0]
                print(Fore.YELLOW + f"\n🏆 {winner.name} wins by default! Collected pot: {self.pot}")
                winner.stack += self.pot
                self.round_over = True
                return

            action_happened = False
            for player in self.players:
                if player.folded or player.stack == 0:
                    continue

                action = player.decide_action(self.current_bet, self.pot, self.community_cards, self.deck)
                if self.verbose:
                    print(Fore.LIGHTBLUE_EX + f"{player.name} decides to {action}")

                if action["action"] == "fold":
                    player.folded = True
                elif action["action"] == "call":
                    self.pot += action["amount"]
                elif action["action"] == "raise":
                    self.pot += action["amount"]
                    self.current_bet = player.bet
                    action_happened = True
                elif action["action"] == "check":
                    pass

            if not action_happened:
                break

    def play_round(self):
        self.reset_for_new_round()
        self.deal_hole_cards()
        if self.verbose:
            for p in self.players:
                print(Fore.MAGENTA + f"{p.name}'s hand: {p.hand}")
        self.betting_round()

        self.deal_community_cards(3)
        self.betting_round()

        self.deal_community_cards(1)
        self.betting_round()

        self.deal_community_cards(1)
        self.betting_round()

        if not self.round_over:
            self.determine_winner()

    def determine_winner(self):
        print(Fore.MAGENTA + "\n--- Round Results ---")
        active = [p for p in self.players if not p.folded]
        if len(active) == 1:
            winner = active[0]
            winner.stack += self.pot
            print(Fore.YELLOW + f"\n🏆 {winner.name} wins by default! Collected pot: {self.pot}")
            return

        hands = [(p, Hand(p.hand, self.community_cards)) for p in active]
        for p, h in hands:
            print(Fore.CYAN + f"{p.name}'s hand: {h}")

        winner, best_hand = max(hands, key=lambda x: x[1])
        winner.stack += self.pot
        print(Fore.GREEN + f"\n🏆 {winner.name} wins the pot of {self.pot} with {best_hand}")
