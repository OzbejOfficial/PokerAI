import random
from hand import Hand

RAISE_AMOUNT = 10


class Player:
    def __init__(self, name, stack=1000):
        self.name = name
        self.stack = stack
        self.hand = []  # 2 hole cards
        self.bet = 0
        self.folded = False

    def reset(self):
        self.hand = []
        self.folded = False
        self.bet = 0

    def reset_for_round(self):
        self.hand = []
        self.bet = 0
        self.folded = False

    def decide_action(self, current_bet, pot, community_cards=None, deck=None):
        raise NotImplementedError("Subclasses must implement decide_action()")


class Human(Player):
    def decide_action(self, current_bet, pot, community_cards=None, deck=None):
        print(f"\n{self.name}, your hand: {self.hand}")
        print(f"Community cards: {community_cards}")
        print(f"Pot: {pot}, Current bet: {current_bet}, Your bet: {self.bet}, Your stack: {self.stack}")
        print("Actions: [0=Fold, 1=Check, 2=Call, 3=Raise]")

        while True:
            try:
                action = int(input("Enter your action number: "))
                if action == 0:
                    self.folded = True
                    return {"action": "fold"}
                elif action == 1 and current_bet == self.bet:
                    return {"action": "check"}
                elif action == 2:
                    call_amount = current_bet - self.bet
                    actual_call = min(self.stack, call_amount)
                    self.stack -= actual_call
                    self.bet += actual_call
                    return {"action": "call", "amount": actual_call}
                elif action == 3:
                    raise_amount = min(self.stack, RAISE_AMOUNT)
                    self.stack -= raise_amount
                    self.bet += raise_amount
                    return {"action": "raise", "amount": raise_amount}
                else:
                    print("Invalid input or can't check when there's a bet.")
            except:
                print("Enter a valid number.")



class RandomBot(Player):
    def __init__(self, name, stack=1000, verbose=False):
        super().__init__(name, stack)
        self.verbose = verbose

    def decide_action(self, current_bet, pot, community_cards=None, deck=None):
        actions = ['fold', 'call', 'raise', 'check'] if current_bet > self.bet else ['check', 'raise']
        action = random.choice(actions)

        if action == 'fold':
            self.folded = True
            if self.verbose:
                print(f"{self.name} FOLDS")
            return {"action": "fold"}
        elif action == 'call':
            call_amount = current_bet - self.bet
            actual_call = min(call_amount, self.stack)
            self.stack -= actual_call
            self.bet += actual_call
            if self.verbose:
                print(f"{self.name} CALLS {actual_call}")
            return {"action": "call", "amount": actual_call}
        elif action == 'raise':
            raise_amount = min(self.stack, RAISE_AMOUNT)
            self.stack -= raise_amount
            self.bet += raise_amount
            if self.verbose:
                print(f"{self.name} RAISES {raise_amount}")
            return {"action": "raise", "amount": raise_amount}
        elif action == 'check':
            if self.verbose:
                print(f"{self.name} CHECKS")
            return {"action": "check"}


class StatisticalBot(Player):
    def __init__(self, name, stack=1000, preflop_raise=0.9, preflop_call=0.5,
                 postflop_raise=0.7, postflop_call=0.4, simulations=100, verbose=False):
        super().__init__(name, stack)
        self.preflop_raise = preflop_raise
        self.preflop_call = preflop_call
        self.postflop_raise = postflop_raise
        self.postflop_call = postflop_call
        self.simulations = simulations
        self.verbose = verbose

    def estimate_win_probability(self, community_cards, deck, num_opponents=1):
        wins = 0
        pool = [card for card in deck.cards if card not in self.hand + community_cards]
        for _ in range(self.simulations):
            random.shuffle(pool)
            opp_hands = [pool[i:i+2] for i in range(0, num_opponents * 2, 2)]
            my_hand = Hand(self.hand, community_cards)
            opp_hands_eval = [Hand(h, community_cards) for h in opp_hands]
            if all(my_hand.compare(opp) >= 0 for opp in opp_hands_eval):
                wins += 1
        return wins / self.simulations

    def decide_action(self, current_bet, pot, community_cards=None, deck=None):
        if community_cards is None or deck is None:
            raise ValueError("StatisticalBot requires community cards and deck")

        win_prob = self.estimate_win_probability(community_cards, deck)
        phase = 'preflop' if len(community_cards) == 0 else 'postflop'
        raise_thresh = getattr(self, f"{phase}_raise")
        call_thresh = getattr(self, f"{phase}_call")

        if win_prob >= raise_thresh:
            amount = min(self.stack, RAISE_AMOUNT)
            self.stack -= amount
            self.bet += amount
            if self.verbose:
                print(f"{self.name} RAISES (win_prob={win_prob:.2f})")
            return {"action": "raise", "amount": amount}
        elif win_prob >= call_thresh:
            call_amt = current_bet - self.bet
            actual_call = min(self.stack, call_amt)
            self.stack -= actual_call
            self.bet += actual_call
            if self.verbose:
                print(f"{self.name} CALLS (win_prob={win_prob:.2f})")
            return {"action": "call", "amount": actual_call}
        else:
            if current_bet > self.bet:
                self.folded = True
                if self.verbose:
                    print(f"{self.name} FOLDS (win_prob={win_prob:.2f})")
                return {"action": "fold"}
            else:
                if self.verbose:
                    print(f"{self.name} CHECKS")
                return {"action": "check"}


class RLBot(Player):
    def __init__(self, name="RLBot", stack=1000):
        super().__init__(name, stack)
        self._action = None  # This is set externally by the RL agent

    def set_action(self, action_dict):
        self._action = action_dict

    def decide_action(self, current_bet, pot, community_cards=None, deck=None):
        if self._action is None:
            raise RuntimeError("RLBot must have an action set before calling decide_action()")
        action = self._action
        self._action = None  # Clear after use

        if action["action"] == "raise":
            raise_amount = min(self.stack, RAISE_AMOUNT)
            self.stack -= raise_amount
            self.bet += raise_amount
            return {"action": "raise", "amount": raise_amount}
        elif action["action"] == "call":
            call_amount = current_bet - self.bet
            actual_call = min(self.stack, call_amount)
            self.stack -= actual_call
            self.bet += actual_call
            return {"action": "call", "amount": actual_call}
        elif action["action"] == "fold":
            self.folded = True
            return {"action": "fold"}
        elif action["action"] == "check":
            return {"action": "check"}
        else:
            raise ValueError("Invalid RLBot action received")
        
    def reset(self):
        super().reset()
        self._action = None

