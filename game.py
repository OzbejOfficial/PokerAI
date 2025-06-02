from deck import Deck
from hand import Hand

from colorama import Fore, Style, init
init(autoreset=True)

class Game:
    def __init__(self, starting_bet=10, verbose=False):
        self.players = []
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.verbose = verbose
        self.starting_bet = starting_bet
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
        if self.round_over:
            return

        print(Fore.MAGENTA + "\n--- Betting Round ---")
        print(Fore.MAGENTA + f"Community Cards: {self.community_cards}")
        if self.verbose:
            for p in self.players:
                print(f"{p.name} stack: {p.stack}, folded: {p.folded}, bet: {p.bet}")

        # Initialize
        active_players = [p for p in self.players if not p.folded and p.stack > 0]
        if len(active_players) == 1:
            winner = active_players[0]
            winner.stack += self.pot
            print(Fore.YELLOW + f"\n🏆 {winner.name} wins by default! Collected pot: {self.pot}")
            self.round_over = True
            return

        # Betting loop
        while True:
            bets_changed = False
            for player in self.players:
                if player.folded or player.stack == 0:
                    continue

                if all(p.bet == self.current_bet or p.folded or p.stack == 0 for p in self.players):
                    continue  # Everyone matched, skip redundant actions

                action = player.decide_action(self.current_bet, self.pot, self.community_cards, self.deck)
                if self.verbose:
                    print(Fore.LIGHTBLUE_EX + f"{player.name} decides to {action}")

                if action["action"] == "fold":
                    player.folded = True
                elif action["action"] == "call":
                    self.pot += action["amount"]
                    bets_changed = True
                elif action["action"] == "raise":
                    self.pot += action["amount"]
                    self.current_bet = player.bet
                    bets_changed = True
                elif action["action"] == "check":
                    continue

            # Check if all active players have equal bets or folded
            active_bets = [p.bet for p in self.players if not p.folded and p.stack > 0]
            if len(set(active_bets)) <= 1 or not bets_changed:
                break

    def play_round(self):
        self.reset_for_new_round()
        self.deal_hole_cards()
        if self.verbose:
            for p in self.players:
                print(Fore.MAGENTA + f"{p.name}'s hand: {p.hand}")

        self.betting_round()
        if self.round_over: return

        self.deal_community_cards(3)
        self.game_log.append(f"FLOP: {self.community_cards}")
        self.betting_round()
        if self.round_over: return

        self.deal_community_cards(1)
        self.game_log.append(f"TURN: {self.community_cards}")
        self.betting_round()
        if self.round_over: return

        self.deal_community_cards(1)
        self.game_log.append(f"RIVER: {self.community_cards}")
        self.betting_round()
        if self.round_over: return

        self.determine_winner()

    def determine_winner(self):
        print(Fore.MAGENTA + "\n--- Round Results ---")
        active = [p for p in self.players if not p.folded]

        self.winners = []

        if len(active) == 1:
            winner = active[0]
            winner.stack += self.pot
            self.winners = [winner]
            print(Fore.YELLOW + f"\n🏆 {winner.name} wins by default! Collected pot: {self.pot}")
            return

        # Evaluate hands
        hands = [(p, Hand(p.hand, self.community_cards)) for p in active]

        for p, h in hands:
            print(Fore.CYAN + f"{p.name}'s hand: {h}")

        # Determine the winning hand
        best_score = max(hands, key=lambda x: x[1])
        best_players = [p for p, h in hands if h == best_score[1]]

        for winner in best_players:
            winner.stack += self.pot // len(best_players)

        self.winners = best_players

        if len(best_players) == 1:
            print(Fore.GREEN + f"\n🏆 {best_players[0].name} wins the pot of {self.pot} with {best_score[1]}")
        else:
            print(Fore.GREEN + f"\n🏆 Tie between {[p.name for p in best_players]}! Split pot: {self.pot}")

