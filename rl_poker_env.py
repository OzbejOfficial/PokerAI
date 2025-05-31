import gym
from gym import spaces
import numpy as np
from game import Game
from player import RLBot, RandomBot
import csv
import os

class RLPokerEnv(gym.Env):
    def __init__(self, num_opponents=1, log_path="logs/poker_log.csv", initial_stack=1000):
        super().__init__()
        self.num_opponents = num_opponents
        self.initial_stack = initial_stack
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._init_log()
        self._build_game()

        self.action_space = spaces.Discrete(4)  # 0: fold, 1: check, 2: call, 3: raise
        self.observation_space = spaces.Dict({
            "agent_stack": spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32),
            "current_bet": spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32),
            "pot": spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32),
            "num_players": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)
        })

        self.done = False
        self.phase = 0
        self.prev_stack = None
        self.current_player_index = 0

    def _init_log(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "stack_delta", "result"])
        self.episode_count = 0

    def _build_game(self):
        self.game = Game(verbose=True)
        self.agent = RLBot("RLAgent", stack=self.initial_stack)
        self.opponents = [RandomBot(f"Bot{i+1}", stack=self.initial_stack) for i in range(self.num_opponents)]
        self.players = [self.agent] + self.opponents
        for p in self.players:
            self.game.add_player(p)

    def reset(self):
        for player in self.players:
            if player.stack <= 0:
                player.stack = self.initial_stack
        self.game.reset_for_new_round()
        self.phase = 0
        self.done = False
        self.prev_stack = self.agent.stack
        self.current_player_index = 0
        return self._get_obs()

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, {}

        player = self.players[self.current_player_index]
        if player.stack == 0 or player.folded:
            self._advance_turn()
            return self._get_obs(), 0, self.done, {}

        if isinstance(player, RLBot):
            player.set_action(self._action_to_dict(action))

        self._play_turn(player)
        self._advance_turn()

        reward = 0
        if self.done:
            reward = self._calculate_reward()
            self._log_episode(reward)

        return self._get_obs(), reward, self.done, {}

    def _play_turn(self, player):
        if player.folded or player.stack == 0:
            return

        action = player.decide_action(self.game.current_bet, self.game.pot, self.game.community_cards, self.game.deck)
        if action["action"] == "fold":
            player.folded = True
        elif action["action"] == "call":
            self.game.pot += action["amount"]
        elif action["action"] == "raise":
            self.game.pot += action["amount"]
            self.game.current_bet = player.bet
        elif action["action"] == "check":
            pass

    def _advance_turn(self):
        active = [p for p in self.players if not p.folded]
        if len(active) == 1:
            self.game.round_over = True
            self._finalize_game()
            self.done = True
            return

        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        while self.players[self.current_player_index].folded or (
            self.players[self.current_player_index].stack == 0 and not isinstance(self.players[self.current_player_index], RLBot)
        ):
            self.current_player_index = (self.current_player_index + 1) % len(self.players)

        # Check if betting round is done
        if all(p.bet == self.game.current_bet or p.folded or p.stack == 0 for p in self.players):
            self.phase += 1
            if self.phase == 1:
                self.game.deal_community_cards(3)
            elif self.phase in [2, 3]:
                self.game.deal_community_cards(1)
            elif self.phase > 3:
                self._finalize_game()
                self.done = True
                return
            self.current_player_index = 0

    def _finalize_game(self):
        if not self.game.round_over:
            self.game.determine_winner()

    def _calculate_reward(self):
        return self.agent.stack - self.prev_stack

    def _log_episode(self, reward):
        result = "win" if reward > 0 else "loss" if reward < 0 else "tie"
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.episode_count, reward, result])
        self.episode_count += 1

    def _action_to_dict(self, action):
        return [
            {"action": "fold"},
            {"action": "check"},
            {"action": "call"},
            {"action": "raise"}
        ][action]

    def _get_obs(self):
        return {
            "agent_stack": np.array([self.agent.stack], dtype=np.float32),
            "current_bet": np.array([self.game.current_bet], dtype=np.float32),
            "pot": np.array([self.game.pot], dtype=np.float32),
            "num_players": np.array([len(self.players)], dtype=np.int32)
        }

    def render(self, mode="human"):
        print(f"--- RL Turn ---")
        print(f"Phase: {self.phase} | Pot: {self.game.pot} | Current Bet: {self.game.current_bet}")
        print(f"Community: {self.game.community_cards}")
        print(f"Your Hand: {self.agent.hand} | Stack: {self.agent.stack}")
        print(f"Opponent Stacks: {[p.stack for p in self.opponents]}")

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]