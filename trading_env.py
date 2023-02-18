import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class Actions(Enum):
    Hold = 0
    Buy = 1
    Sell = 2


class Positions(Enum):
    Flat = 0
    Long = 1
    Short = 2
    

class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size):
        assert df.ndim == 2

        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._action_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None


    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._start_tick
        self._position = Positions.Flat
        self._action_history = (self.window_size * [None]) + [0]
        self._total_reward = 0.
        self._total_profit = 100.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()


    def step(self, action):
        self._done = False
        self._current_tick += 1
        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)


            
        if action == Actions.Hold.value:
            self._action_history.append(0)
            
        if action == Actions.Buy.value:
            self._action_history.append(1)
            if self._position == Positions.Flat:
                self._position = Positions.Long
                self._last_trade_tick = self._current_tick
            if self._position == Positions.Long:
                self._position = Positions.Long
                self._last_trade_tick = self._current_tick
            if self._position == Positions.Short:
                self._position = Positions.Flat
                
        if action == Actions.Sell.value:
            self._action_history.append(2)
            if self._position == Positions.Flat:
                self._position = Positions.Short
                self._last_trade_tick = self._current_tick
            if self._position == Positions.Short:
                self._position = Positions.Short
                self._last_trade_tick = self._current_tick
            if self._position == Positions.Long:
                self._position = Positions.Flat
        
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info


    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]


    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._action_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)


    def render_all(self, mode='human'):
        # window_ticks = np.arange(len(self._action_history))
        plt.plot(self.prices)

        for i in range(len(self._action_history)):
            if self._action_history[i] == None:
                continue
            # elif self._action_history[i] == 0:
            #     plt.plot(i, self.prices[i], 'yo')
            elif self._action_history[i] == 1:
                plt.plot(i, self.prices[i], 'go')
            elif self._action_history[i] == 2:
                plt.plot(i, self.prices[i], 'ro')


        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
        
        
    def close(self):
        plt.close()


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()


    def _process_data(self):
        raise NotImplementedError


    def _calculate_reward(self, action):
        raise NotImplementedError


    def _update_profit(self, action):
        raise NotImplementedError


    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError