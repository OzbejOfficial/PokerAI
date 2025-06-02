import gym
from gym import spaces
import numpy as np
import csv
import os
from card import Card
from deck import Deck
from hand import Hand
from player import RLBot, RandomBot, StatisticalBot

class RLPokerEnv(gym.Env):
    def __init__(self, initial_stack=1000, num_opponents=1, log_path="logs/poker_log.csv"):
        super().__init__()
        self.initial_stack = initial_stack
        self.num_opponents = num_opponents
        self.log_path = log_path

        self.agent = RLBot("RLAgent", stack=initial_stack)
        self.opponents = [StatisticalBot(f"StatBot{i}", stack=initial_stack) for i in range(num_opponents)]

        self.players = [self.agent] + self.opponents
        self.deck = Deck()
        self.starting_bet = 10

        self.observation_space = spaces.Box(low=0, high=1, shape=(107,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._init_logger()
        self.reset()

    def _init_logger(self):
        self.logfile = open(self.log_path, "w", newline="")
        self.logger = csv.writer(self.logfile)
        self.logger.writerow(["Episode", "Phase", "Player", "Action", "Amount", "Pot", "Stack", "Community", "Hole"])
        self.episode_counter = 0

    def log_action(self, phase, player, action, amount):
        try:
            self.logger.writerow([
                self.episode_counter,
                phase,
                player.name,
                action,
                amount,
                self.pot,
                player.stack,
                ";".join(str(c) for c in self.community_cards),
                ";".join(str(c) for c in player.hand)
            ])
        except Exception as e:
            print("Logging error:", e)

    def reset(self):
        self.episode_counter += 1
        for player in self.players:
            if player.stack <= self.initial_stack * 0.1 or player.stack > 3 * self.initial_stack:
                player.stack = self.initial_stack

        self.deck.reset()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.done = False
        for p in self.players:
            p.folded = False
            p.bet = self.starting_bet if p.stack >= self.starting_bet else 0
            p.stack -= p.bet
            self.pot += p.bet
            p.hand = []

        for player in self.players:
            player.hand = self.deck.deal(2)

        self.community_cards = self.deck.deal(3)
        return self._get_obs()

    def step(self, action_sequence):
        assert len(action_sequence) == 3, "Must provide 3 action policies (flop, turn, river)"

        for i, phase in enumerate(['flop', 'turn', 'river']):
            self.current_bet = 0
            for p in self.players:
                p.bet = 0

            if phase == 'turn':
                self.community_cards += self.deck.deal(1)
            elif phase == 'river':
                self.community_cards += self.deck.deal(1)

            self._betting_round(policy=action_sequence[i], phase=phase)
            if self.agent.folded:
                self.done = True
                return self._get_obs(), -self.pot, self.done, {}

        reward = self._determine_winner()
        self.done = True
        return self._get_obs(), reward, self.done, {}

    def _betting_round(self, policy, phase):
        max_iterations = 100
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            changes_made = False
            for player in self.players:
                if player.stack == 0 or player.folded:
                    continue

                if isinstance(player, RLBot):
                    action = self._decide_with_policy(policy, player)
                    if action["action"] in ["call", "raise"] and ("amount" not in action or action["amount"] <= 0):
                        action = {"action": "fold"}
                    if action["action"] == "raise" and action["amount"] + self.current_bet <= player.bet:
                        action = {"action": "call", "amount": self.current_bet - player.bet}
                    player.set_action(action)
                    decision = player.decide_action(self.current_bet, self.pot, self.community_cards, self.deck)
                else:
                    decision = player.decide_action(self.current_bet, self.pot, self.community_cards, self.deck)

                if decision["action"] == "fold":
                    player.folded = True
                    self.log_action(phase, player, "fold", 0)
                elif decision["action"] == "call":
                    actual_call = min(decision["amount"], player.stack, self.current_bet - player.bet)
                    self.pot += actual_call
                    player.stack -= actual_call
                    player.bet += actual_call
                    self.log_action(phase, player, "call", actual_call)
                    changes_made = True
                elif decision["action"] == "raise":
                    actual_raise = min(decision["amount"], player.stack)
                    if actual_raise > 0:
                        self.pot += actual_raise
                        player.stack -= actual_raise
                        player.bet += actual_raise
                        self.current_bet = player.bet
                        self.log_action(phase, player, "raise", actual_raise)
                        changes_made = True
                elif decision["action"] == "check":
                    self.log_action(phase, player, "check", 0)

            if all(p.folded or p.bet == self.current_bet or p.stack == 0 for p in self.players):
                if not changes_made:
                    break

    def _decide_with_policy(self, policy, player):
        call_amt = self.current_bet - player.bet
        stack = player.stack

        if call_amt == 0:
            if policy['wanted_action'] == 'check':
                return {"action": "check"}
            elif policy['wanted_action'] == 'raise' and stack >= policy['raise_amount']:
                return {"action": "raise", "amount": policy['raise_amount']}
            else:
                return {"action": "check"}

        if call_amt <= policy['call_till'] and call_amt <= stack:
            if policy['wanted_action'] == 'call':
                return {"action": "call", "amount": call_amt}
            elif policy['wanted_action'] == 'raise' and stack >= policy['raise_amount']:
                return {"action": "raise", "amount": policy['raise_amount']}
        else:
            if policy['action_vs_raise'] == 'fold':
                return {"action": "fold"}
            elif policy['action_vs_raise'] == 'call' and call_amt <= stack:
                return {"action": "call", "amount": call_amt}
            elif policy['action_vs_raise'] == 'reraise' and stack >= policy['reraise_amount']:
                return {"action": "raise", "amount": policy['reraise_amount']}

        return {"action": "fold"}

    def _determine_winner(self):
        active = [p for p in self.players if not p.folded]
        if len(active) == 1:
            winner = active[0]
            winner.stack += self.pot
            self.log_action("showdown", winner, "wins_by_fold", self.pot)
            self.last_winner = winner
            return self.pot if winner == self.agent else -self.pot

        hands = [(p, Hand(p.hand, self.community_cards)) for p in active]
        hands.sort(key=lambda x: x[1])
        best_score = hands[0][1]
        winners = [p for p, h in hands if h == best_score]

        pot_share = self.pot // len(winners)
        for winner in winners:
            winner.stack += pot_share

        for p in self.players:
            self.log_action("showdown", p, "hand", 0)

        self.last_winner = winners[0] if len(winners) == 1 else None
        return pot_share if self.agent in winners else -self.pot

    def _one_hot_cards(self, cards):
        vec = np.zeros(52)
        for card in cards:
            suit_idx = Card.suits.index(card.suit)
            rank_idx = Card.ranks.index(card.rank)
            idx = suit_idx * 13 + rank_idx
            vec[idx] = 1
        return vec

    def _get_obs(self):
        hole_vec = self._one_hot_cards(self.agent.hand)
        board_vec = self._one_hot_cards(self.community_cards)
        stack = np.array([self.agent.stack / 1000], dtype=np.float32)
        pot = np.array([self.pot / 1000], dtype=np.float32)
        strength = np.array([1 - Hand(self.agent.hand, self.community_cards).score / 7462], dtype=np.float32)
        return np.concatenate([hole_vec, board_vec, pot, stack, strength])

    def render(self, mode='human'):
        print(f"Pot: {self.pot}")
        print(f"Community: {self.community_cards}")
        print(f"Agent: {self.agent.hand}, stack: {self.agent.stack}")
        for opp in self.opponents:
            print(f"{opp.name}: {opp.hand}, stack: {opp.stack}")
        if hasattr(self, "last_winner"):
            print("Winner:", self.last_winner.name if self.last_winner else "Tie")
