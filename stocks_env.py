import numpy as np

from trading_env import TradingEnv, Actions, Positions


class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size)

        self.trade_fee_bid_percent = 0.001  # unit
        self.trade_fee_ask_percent = 0.001  # unit


    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features


    def _calculate_reward(self, action):
        step_reward = 0
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        price_diff = current_price - last_trade_price
        fee = self._total_profit * 0.001
        
        if action == Actions.Hold.value:
            if self._position == Positions.Flat:
                step_reward = 0
            if self._position == Positions.Long:
                if price_diff > 0:
                    step_reward = 0
                if price_diff < 0:
                    step_reward -= 0.1
            if self._position == Positions.Short:
                if price_diff > 0:
                    step_reward -= 0.1
                if price_diff < 0:
                    step_reward = 0
        if action == Actions.Buy.value:
            if self._position == Positions.Flat:
                step_reward -= fee
            if self._position == Positions.Long:
                step_reward -= fee
                if price_diff > 0:
                    step_reward += abs(price_diff)/last_trade_price
                if price_diff < 0:
                    step_reward -= abs(price_diff)/last_trade_price
            if self._position == Positions.Short:
                step_reward -= fee
                if price_diff > 0:
                    step_reward -= abs(price_diff)/last_trade_price
                if price_diff < 0:
                    step_reward += abs(price_diff)/last_trade_price
        if action == Actions.Sell.value:
            if self._position == Positions.Flat:
                step_reward -= fee
            if self._position == Positions.Long:
                step_reward -= fee
                if price_diff > 0:
                    step_reward += abs(price_diff)/last_trade_price
                if price_diff < 0:
                    step_reward -= abs(price_diff)/last_trade_price
            if self._position == Positions.Short:
                step_reward -= fee
                if price_diff > 0:
                    step_reward -= abs(price_diff)/last_trade_price
                if price_diff < 0:
                    step_reward += abs(price_diff)/last_trade_price
                
            

        return step_reward


    def _update_profit(self, action):
        fee = self._total_profit * 0.001
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        price_diff = current_price - last_trade_price
        
        
        if action == Actions.Buy.value:
            if self._position == Positions.Flat:
                self._total_profit = self._total_profit - fee
            if self._position == Positions.Short:
                self._total_profit = self._total_profit - fee
                if price_diff < 0:
                    self._total_profit = self._total_profit + abs(price_diff)/last_trade_price
                if price_diff > 0:
                    self._total_profit = self._total_profit - abs(price_diff)/last_trade_price
            if self._position == Positions.Long:
                self._total_profit = self._total_profit - fee
                if price_diff > 0:
                    self._total_profit = self._total_profit + abs(price_diff)/last_trade_price
                if price_diff < 0:
                    self._total_profit = self._total_profit - abs(price_diff)/last_trade_price
        if action == Actions.Sell.value:
            if self._position == Positions.Flat:
                self._total_profit = self._total_profit - fee
            if self._position == Positions.Long:
                self._total_profit = self._total_profit - fee
                if price_diff > 0:
                    self._total_profit = self._total_profit + abs(price_diff)/last_trade_price
                if price_diff < 0:
                    self._total_profit = self._total_profit - abs(price_diff)/last_trade_price
            if self._position == Positions.Short:
                self._total_profit = self._total_profit - fee
                if price_diff < 0:
                    self._total_profit = self._total_profit + abs(price_diff)/last_trade_price
                if price_diff > 0:
                    self._total_profit = self._total_profit - abs(price_diff)/last_trade_price

                


    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit