import gym
from gym import spaces
import numpy as np
import os
import csv

from game import Game
from player import RLBot, RandomBot
from hand import Hand
from card import Card

class RLPokerEnv(gym.Env):
    def __init__(self, num_opponents=1, initial_stack=1000, log_path="logs/poker_log.csv", starting_bet=10):
        super().__init__()
        self.num_opponents = num_opponents
        self.initial_stack = initial_stack
        self.starting_bet = starting_bet
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        self.action_space = spaces.Discrete(4)  # 0: Fold, 1: Check, 2: Call, 3: Raise

        # Observation format for RecurrentPPO (use dict with mask)
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0, high=1e5, shape=(4,), dtype=np.float32),
            "mask": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        })

        self._build_game()
        self._init_log()
        self.episode_count = 0
        self.done = False

    def _build_game(self):
        self.game = Game(starting_bet=self.starting_bet, verbose=False)
        self.agent = RLBot("RLAgent", stack=self.initial_stack)
        self.opponents = [RandomBot(f"Bot{i+1}", stack=self.initial_stack) for i in range(self.num_opponents)]
        self.players = [self.agent] + self.opponents
        for p in self.players:
            self.game.add_player(p)

    def _init_log(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward", "result"])

    def reset(self):
        for player in self.players:
            if player.stack <= 0:
                player.stack = self.initial_stack
            player.reset()

        self.game.reset_for_new_round()
        self.game.deal_hole_cards()
        self.phase = 0
        self.current_player_index = 0
        self.done = False
        self.awaiting_rl_action = False
        self.reward = 0
        self.agent_action_log = []

        self.phases = [
            ("FLOP", 3),
            ("TURN", 1),
            ("RIVER", 1)
        ]

        self._start_next_phase()
        return self._get_obs(), {}

    def _start_next_phase(self):
        if self.phase >= len(self.phases):
            self._finalize_round()
            return
        _, n_cards = self.phases[self.phase]
        self.game.deal_community_cards(n_cards)
        self.phase += 1
        self.current_player_index = 0
        self.acted = set()
        self.game.current_bet = 0
        for p in self.players:
            p.bet = 0
        self._continue_phase()

    def _continue_phase(self):
        while True:
            remaining = [p for p in self.players if not p.folded and p.stack > 0]
            if len(remaining) <= 1:
                self.done = True
                self._finalize_round()
                return

            player = self.players[self.current_player_index]
            if player.folded or player.stack <= 0:
                self.current_player_index = (self.current_player_index + 1) % len(self.players)
                continue

            if isinstance(player, RLBot):
                self.awaiting_rl_action = True
                return

            action = player.decide_action(self.game.current_bet, self.game.pot, self.game.community_cards, self.game.deck)
            self._apply_action(player, action)
            self.acted.add(player.name)

            if len([p for p in self.players if not p.folded and p.stack > 0]) <= 1:
                self.done = True
                self._finalize_round()
                return

            if all(p.name in self.acted or p.folded or p.stack == 0 for p in self.players):
                self.awaiting_rl_action = False
                self._start_next_phase()
                return

            self.current_player_index = (self.current_player_index + 1) % len(self.players)

    def _apply_action(self, player, action):
        if action["action"] == "fold":
            player.folded = True
        elif action["action"] == "check":
            pass
        elif action["action"] == "call":
            call_amt = min(self.game.current_bet - player.bet, player.stack)
            player.stack -= call_amt
            player.bet += call_amt
            self.game.pot += call_amt
        elif action["action"] == "raise":
            amt = min(action.get("amount", 10), player.stack)
            player.stack -= amt
            player.bet += amt
            self.game.pot += amt
            self.game.current_bet = player.bet
            self.acted = set()

    def step(self, action):
        if self.done:
            return self._get_obs(), self.reward, True, False, {}

        action_dict = self._action_to_dict(action)
        self._apply_action(self.agent, action_dict)
        self.acted.add(self.agent.name)
        self.agent_action_log.append(action_dict)

        remaining = [p for p in self.players if not p.folded and p.stack > 0]
        if len(remaining) <= 1:
            self.done = True
            self._finalize_round()
        else:
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
            self._continue_phase()

        return self._get_obs(), self.reward, self.done, False, {}

    def _finalize_round(self):
        self.done = True
        win = self._is_better()
        self.reward = self.agent.bet if win else -self.agent.bet
        result = "win" if win else "loss"

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([self.episode_count, self.reward, result])
        self.episode_count += 1

    def _is_better(self):
        round_players = [p for p in self.players if not p.folded]
        if len(round_players) == 1:
            return self.agent in round_players

        agent_hand = Hand(self.agent.hand, self.game.community_cards)
        for p in round_players:
            if p == self.agent:
                continue
            if Hand(p.hand, self.game.community_cards) > agent_hand:
                return False
        return True

    def _get_obs(self):
        return {
            "obs": np.array([
                self.agent.stack,
                self.game.current_bet,
                self.game.pot,
                len([p for p in self.players if not p.folded])
            ], dtype=np.float32),
            "mask": np.array([1, 1, 1, 1], dtype=np.float32)
        }

    def _action_to_dict(self, action):
        return [
            {"action": "fold"},
            {"action": "check"},
            {"action": "call"},
            {"action": "raise", "amount": 10}
        ][action]
