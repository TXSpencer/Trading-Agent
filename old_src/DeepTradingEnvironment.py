
class DeepTradingEnvironment:

    def __init__(self, data, window_size, initial_cash=1000):
        self.data = data
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.current_step = 0
        self.cash = initial_cash
        self.shares_held = 0
        self.total_profit = 0
        self.done = False


    def reset(self):
        """
        Reset the environment for the beginning of a new episode
        """
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.total_profit = 0
        self.done = False

        return self._get_state()
    

    def _get_state(self):
        """
        Return the actual state as a sliding window
        """
        start = max(0, self.current_step - self.window_size)
        end = self.current_step
        return self.data[start:end].values
    
    
    def step(self, action):
        """
        Execute an action and return (next_state, reward, done)
        """
        if self.done:
            return self._get_state(), 0, self.done

        current_price = self.data.iloc[self.current_step]['Close']

        # Execute the action (0=Hold, 1=Buy, 2=Sell)
        reward = 0

        if action == 1:
            
            num_shares = self.cash // current_price
            self.shares_held += num_shares
            self.cash -= num_shares * current_price
        
        elif action == 2 and self.shares_held > 0:

            self.cash += self.shares_held * current_price
            reward = self.cash - self.initial_cash
            self.shares_held = 0
        
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1
        
        return self._get_state(), reward, self.done
            