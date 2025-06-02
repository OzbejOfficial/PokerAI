import gym
from gym import spaces
import numpy as np
import random
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
        self.combined_rewards = 0
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        for player in self.players:
            player.reset()

        # Reset player stacks if needed
        for player in self.players:
            if player.stack <= 0 or player.stack > 3000:
                player.stack = self.initial_stack

        self._play_round()

        obs = self._get_obs()
        info = {}
        return obs, info


    def _play_round(self):
        self.game_log = []
        self.done = False
        self.phase = 0
        self.awaiting_rl_action = False
        self.game.reset_for_new_round()

        self.game.deal_hole_cards()
        self.game_log.append("DEALT CARDS.")
        for player in self.players:
            self.game_log.append(f"{player.name} has {player.hand}")

        # self.phase = 1
        self.game_log.append("NO PREFLOP.")
        # self.current_player_index = 0
        # self._play_phase()
        # if self.done:
        #     self._finalize_round()
        #     return

        self.phase = 2
        self.game.deal_community_cards(3)
        self.game_log.append(f"FLOP: {self.game.community_cards}")
        self.current_player_index = 0
        self.start_phase()
        if self.done:
            self._finalize_round()
            return

        self.phase = 3
        self.game.deal_community_cards(1)
        self.game_log.append(f"TURN: {self.game.community_cards}")
        self.current_player_index = 0
        self.start_phase()
        if self.done:
            self._finalize_round()
            return

        self.phase = 4
        self.game.deal_community_cards(1)
        self.game_log.append(f"RIVER: {self.game.community_cards}")
        self.current_player_index = 0
        self.start_phase()
        if self.done:
            self._finalize_round()
            return

        self._finalize_round()

    def start_phase(self):
        for p in self.players:
            p.bet = 0
        self.game.current_bet = 0
        self.acted = set()
        self.last_raiser = None
        self.awaiting_rl_action = False
        self._continue_phase()

    def _continue_phase(self):
        while True:
            remaining = [p for p in self.players if not p.folded and p.stack > 0]
            if len(remaining) <= 1:
                print("Only one player remaining, ending phase.")
                self.done = True
                return

            player = self.players[self.current_player_index]

            if player.folded or player.stack <= 0:
                print(f"{player.name} is folded or out of stack, skipping turn.")
                self.current_player_index = (self.current_player_index + 1) % len(self.players)
                continue

            if isinstance(player, RLBot):
                self.awaiting_rl_action = True
                return  # Step must be called next

            # Bot action
            action = player.decide_action(
                self.game.current_bet,
                self.game.pot,
                self.game.community_cards,
                self.game.deck
            )
            print(f"{player.name} action: {action}")
            self.game_log.append(f"{player.name} action: {action}")

            if action["action"] == "fold":
                player.folded = True
            elif action["action"] == "check":
                pass
            elif action["action"] == "call":
                to_call = min(self.game.current_bet - player.bet, player.stack)
                player.stack -= to_call
                player.bet += to_call
                self.game.pot += to_call
            elif action["action"] == "raise":
                raise_amt = min(action.get("amount", 10), player.stack)
                player.stack -= raise_amt
                player.bet += raise_amt
                self.game.pot += raise_amt
                self.game.current_bet = player.bet
                self.acted = set()  # Reset for new raises
            else:
                raise ValueError(f"Unknown action: {action}")

            self.acted.add(player.name)

            if len([p for p in self.players if not p.folded and p.stack > 0]) <= 1:
                self.done = True
                return

            if all(p.name in self.acted or p.folded or p.stack == 0 for p in self.players):
                break

            self.current_player_index = (self.current_player_index + 1) % len(self.players)

        self.awaiting_rl_action = False


    def step(self, action):
        print(f"Step called with action: {action}")
        print(f"Step and done: {self.done}")

        if self.done:
            reward = self.agent.bet if self._is_better() else -self.agent.bet
            self.combined_rewards += reward
            self.awaiting_rl_action = False
            return self._get_obs(), reward, False, False, {}

        action_dict = self._action_to_dict(action)
        self.game_log.append(f"(RLAgent's turn) Action: {action_dict['action']}")

        if action_dict["action"] == "fold":
            self.agent.folded = True
            self.done = True
            self.game.determine_winner()
        elif action_dict["action"] == "check":
            pass
        elif action_dict["action"] == "call":
            to_call = min(self.game.current_bet - self.agent.bet, self.agent.stack)
            self.agent.stack -= to_call
            self.agent.bet += to_call
            self.game.pot += to_call
        elif action_dict["action"] == "raise":
            raise_amt = min(action_dict.get("amount", 10), self.agent.stack)
            self.agent.stack -= raise_amt
            self.agent.bet += raise_amt
            self.game.pot += raise_amt
            self.game.current_bet = self.agent.bet
            self.acted = set()
        else:
            raise ValueError("Invalid RL action")

        self.acted.add(self.agent.name)

        remaining = [p for p in self.players if not p.folded and p.stack > 0]
        if len(remaining) <= 1:
            self.done = True
            self.game.determine_winner()
        else:
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
            self._continue_phase()

        if self.done:
            self._finalize_round()

        reward = self.agent.bet if self._is_better() else -self.agent.bet
        self.combined_rewards += reward
        return self._get_obs(), reward, False, False, {}



    def _is_better(self):
        round_players = [p for p in self.players if not p.folded]
        if len(round_players) == 1:
            return True
        
        agent_hand = Hand(self.agent.hand, self.game.community_cards)
        for player in round_players[1:]:
            player_hand = Hand(player.hand, self.game.community_cards)
            if player_hand > agent_hand:
                return False
        
        return True

    def _finalize_round(self):
        self.done = True

        # Refill broke players before next round
        for player in self.players:
            if player.stack <= 0:
                player.stack = self.initial_stack

        self.community_card_state = [str(c) for c in self.game.community_cards]
        self.agent_hand_state = [str(c) for c in self.agent.hand]
        self.opponent_hand_state = [[str(c) for c in p.hand] for p in self.opponents]

        self.game.determine_winner()

        self._log_episode(self.combined_rewards)

        self.reset()


    def _log_episode(self, reward):
        if len(self.game.winners) == 0:
            result = "no_winner"
        elif len(self.game.winners) == 1:
            if self.game.winners[0] == self.agent:
                result = "win"
            else:
                result = "loss"
        elif self.agent in self.game.winners:
            result = "tie"
        else:
            result = "loss"

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_count,
                reward,
                result,
                str(self.community_card_state),
                str(self.agent_hand_state),
                str(self.opponent_hand_state),
                str(self.game_log)
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
