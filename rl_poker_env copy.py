import gym
from gym import spaces
import numpy as np
import random
import os
import csv

from game import Game
from player import RLBot, RandomBot

class RLPokerEnv(gym.Env):

    def __init__(self, num_opponents=1, initial_stack=1000, log_path="logs/poker_log.csv", starting_bet=10):
        super().__init__()
        self.num_opponents = num_opponents
        self.initial_stack = initial_stack
        self.starting_bet = starting_bet
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        self.action_space = spaces.Discrete(4)  # 0: Fold, 1: Check, 2: Call, 3: Raise
        self.observation_space = spaces.Dict({
            "agent_stack": spaces.Box(low=0, high=1e5, shape=(1,), dtype=np.float32),
            "current_bet": spaces.Box(low=0, high=1e5, shape=(1,), dtype=np.float32),
            "pot": spaces.Box(low=0, high=1e5, shape=(1,), dtype=np.float32),
            "num_players": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)
        })

        self.episode_count = 0
        self._build_game()
        self._init_log()
        self.game_log = []
        self.action_history = []
        self.awaiting_rl_action = False

    def _build_game(self):
        self.game = Game(starting_bet=self.starting_bet, verbose=True)
        self.agent = RLBot("RLAgent", stack=self.initial_stack)
        self.opponents = [RandomBot(f"Bot{i+1}", stack=self.initial_stack) for i in range(self.num_opponents)]
        self.players = [self.agent] + self.opponents
        for p in self.players:
            self.game.add_player(p)

    def _init_log(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "stack_delta", "result", "community", "agent_hand", "opponent_hands", "game_log"])

    def _play_round(self):
        self.game_log = []
        self.action_history = []
        self.done = False
        self.phase = 0
        self.awaiting_rl_action = False
        self.game.reset_for_new_round()

        self.game.deal_hole_cards()
        self.game_log.append("DEALT CARDS.")
        for player in self.players:
            self.game_log.append(f"{player.name} has {player.hand}")

        self.phase = 1
        self.game_log.append("PREFLOP.")
        self.current_player_index = 0
        self._play_phase()
        if self.done:
            self._finalize_round()
            return

        self.phase = 2
        self.game.deal_community_cards(3)
        self.game_log.append(f"FLOP: {self.game.community_cards}")
        self.current_player_index = 0
        self._play_phase()
        if self.done:
            self._finalize_round()
            return
        
        self.phase = 3
        self.game.deal_community_cards(1)
        self.game_log.append(f"TURN: {self.game.community_cards}")
        self.current_player_index = 0
        self._play_phase()
        if self.done:
            self._finalize_round()
            return
        
        self.phase = 4
        self.game.deal_community_cards(1)
        self.game_log.append(f"RIVER: {self.game.community_cards}")
        self.current_player_index = 0
        self._play_phase()
        if self.done:
            self._finalize_round()
            return

        self._finalize_round()

    def _play_phase(self):
        for p in self.players:
            p.bet = 0
        self.game.current_bet = 0

        active = [p for p in self.players if not p.folded and p.stack > 0]
        acted = set()
        last_raiser = None

        while True:
            player = self.players[self.current_player_index]
            if player.folded or player.stack <= 0:
                self.current_player_index = (self.current_player_index + 1) % len(self.players)
                continue

            if isinstance(player, RLBot):
                self.awaiting_rl_action = True
                return  # Pause here until step() provides action

            action = player.decide_action(
                self.game.current_bet,
                self.game.pot,
                self.game.community_cards,
                self.game.deck
            )
            self.game_log.append(f"{player.name} action: {action}")
            self.action_history.append((player.name, action["action"]))

            if action["action"] == "fold":
                player.folded = True
            elif action["action"] == "check":
                pass
            elif action["action"] == "call":
                amount = min(self.game.current_bet - player.bet, player.stack)
                player.stack -= amount
                player.bet += amount
                self.game.pot += amount
            elif action["action"] == "raise":
                amount = min(action.get("amount", 10), player.stack)
                player.stack -= amount
                player.bet += amount
                self.game.pot += amount
                self.game.current_bet = player.bet
                last_raiser = player
                acted = set()
            else:
                raise ValueError(f"Unknown action: {action}")

            acted.add(player.name)

            remaining = [p for p in self.players if not p.folded and p.stack > 0]
            if len(remaining) == 1:
                self.done = True
                self.game_log.append(f"{remaining[0].name} wins by default!")
                return

            if all(p.name in acted for p in remaining):
                break

            self.current_player_index = (self.current_player_index + 1) % len(self.players)

    def step(self, action):
        assert not self.done, "Step called but game is done."

        action_dict = self._action_to_dict(action)
        self.game_log.append(f"(RLAgent's turn) Action: {action_dict['action']}")
        self.action_history.append(("RLAgent", action_dict['action']))

        if action_dict["action"] == "fold":
            self.agent.folded = True
            self.done = True
        elif action_dict["action"] == "check":
            pass
        elif action_dict["action"] == "call":
            amount = min(self.game.current_bet - self.agent.bet, self.agent.stack)
            self.agent.stack -= amount
            self.agent.bet += amount
            self.game.pot += amount
        elif action_dict["action"] == "raise":
            amount = min(action_dict.get("amount", 10), self.agent.stack)
            self.agent.stack -= amount
            self.agent.bet += amount
            self.game.pot += amount
            self.game.current_bet = self.agent.bet
        else:
            raise ValueError("Invalid action")

        self.awaiting_rl_action = False
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self._play_phase()

        if self.done:
            self._finalize_round()

        reward = float(self.agent.stack - self.initial_stack)
        return self._get_obs(), reward, self.done, {}

    def _finalize_round(self):
        self.game.determine_winner()
        winners = self.game.winners
        reward = 0

        if self.agent in winners:
            split = self.game.pot // len(winners)
            reward = split - self.starting_bet
        elif self.agent.folded:
            reward = -self.starting_bet
        else:
            reward = -self.starting_bet

        self.community_card_state = [str(c) for c in self.game.community_cards]
        self.agent_hand_state = [str(c) for c in self.agent.hand]
        self.opponent_hand_state = [[str(c) for c in p.hand] for p in self.opponents]

        self._log_episode(reward)
        self.done = True

    def _log_episode(self, reward):
        result = "win" if reward > 0 else "loss" if reward < 0 else "tie"
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_count,
                reward,
                result,
                str(self.community_card_state),
                str(self.agent_hand_state),
                str(self.opponent_hand_state),
                str(self.action_history)
            ])
        self.episode_count += 1

    def _action_to_dict(self, action):
        return [
            {"action": "fold"},
            {"action": "check"},
            {"action": "call"},
            {"action": "raise", "amount": 10}
        ][action]

    def _get_obs(self):
        return {
            "agent_stack": np.array([self.agent.stack], dtype=np.float32),
            "current_bet": np.array([self.game.current_bet], dtype=np.float32),
            "pot": np.array([self.game.pot], dtype=np.float32),
            "num_players": np.array([len(self.players)], dtype=np.int32)
        }

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
