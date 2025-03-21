class TradingEnvironment:

    def __init__(self, df):
        
        self.df = df
        self.current_step = 0
        self.balance = 1000
        self.shares = 0
        self.last_buy_price = None

    def step(self, action):
        """
        Simulates a trading day based on the chosen action
        """
        price = self.df.iloc[self.current_step]['Close']
        reward = 0

        if action == 'Buy' and self.balance >= price:
            
            self.shares = self.balance / price
            self.balance = 0
            self.last_buy_price = price
        
        elif action == 'Sell' and self.shares > 0:
            
            self.balance = self.shares * price                          # sell action
            reward = self.balance - (self.shares * self.last_buy_price) # profit generated
            self.shares = 0
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return reward, done