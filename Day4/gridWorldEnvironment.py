import numpy as np                              # type: ignore
import pandas as pd                             # type: ignore
import seaborn                                  # type: ignore
from matplotlib.colors import ListedColormap    # type: ignore

class GridWorld:
    def __init__(self, gamma = 1.0, theta = 0.5, gridSize=4, rewardSize=-1):
        self.actions = ("U", "D", "L", "R") 
        self.states  = np.arange(1, 15)
        
        self.transitions = pd.read_csv("gw.txt", 
                                       header = None, 
                                       sep = "\t",
                                       names=['current_state', 'action', 'next_state', 'reward']).values
                
        self.gamma      = gamma
        self.theta      = theta
        self.gridSize   = gridSize
        self.rewardSize = rewardSize
        
    def state_transition(self, state, action):
        """
        this function returns next state and reward 
        """
        next_state, reward = None, None
        
        for tr in self.transitions:
            if tr[0] == state and tr[1] == action:
                next_state = tr[2]
                reward     = tr[3]
                
        return next_state, reward

    def show_environment(self):
        '''
        Displays the gridworld
        '''
        
        all_states = np.concatenate(([0], self.states, [0])).reshape(4,4)
        colors = []
        
        # colors = ["#ffffff"]
        for i in range(len(self.states) + 1):
            if i == 0:
                colors.append("#c4c4c4")
            else:
                colors.append("#ffffff")

        cmap = ListedColormap(seaborn.color_palette(colors).as_hex())
        
        ax = seaborn.heatmap(all_states, cmap = cmap, \
                             annot = True, linecolor = "#282828", linewidths = 0.2, cbar = False)
    
    def get_policy(self, V):
        all_dirs = []
        all_dirs_chars = []

        for i in range(4):
            for j in range(4):

                neighbors_val = []

                if (i,j) in [(0,0), (3,3)]:
                    dir = ' '
                    dir_char = ' '
                else:
                    # L, R, U, D
                    neighbors = [
                        (i, 0 if j-1<0 else j-1), 
                        (i, 3 if j+1 > 3 else j+1), 
                        (i if i-1 < 0 else i-1, j), 
                        (3 if i+1>3 else i+1, j)
                        ]

                    for each in neighbors:
                        neighbors_val.append(V[each])

                    max_idx = np.argmax(neighbors_val)

                    if max_idx ==0:         # L
                        dir = u'\u2190'
                        dir_char = 'L'
                    elif max_idx ==1:       # R
                        dir = '\u2192'
                        dir_char = 'R'
                    elif max_idx ==2:       # U
                        dir = u'\u2191'
                        dir_char = 'U'
                    else:
                        dir = u'\u2193'     # D
                        dir_char = 'D'
                    
                all_dirs.append(dir)
                all_dirs_chars.append(dir_char)

        all_state_dirs = np.array(all_dirs).reshape(4, 4)

        # combining text with values
        formatted_text = (np.asarray(["{0}\n{1:.6f}".format(text, data) for text, data in zip(all_state_dirs.flatten(), V.flatten())])).reshape(4,4)

        return formatted_text, all_dirs_chars