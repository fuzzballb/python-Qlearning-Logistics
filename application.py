# Artificial Intelligence for business
# Optimizing


# Importing the libraries
import numpy as np
import json

# API with flask
from flask import request
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
     # Setting the parameters gamma and alpha for the Q-Learning
    gamma = 0.75
    alpha = 0.9
     
     
    # Part one - defining the environment
     
    # defining the states
    location_to_state = {'A': 0,
                          'B': 1,
                          'C': 2,
                          'D': 3,
                          'E': 4,
                          'F': 5,
                          'G': 6,
                          'H': 7,
                          'I': 8,
                          'J': 9,
                          'K': 10}
     
     
    # defining the actions
    actions = [0,1,2,3,4,5,6,7,8,9,10]
     
    # defining the rewards
    # rows are states
    # columns are actions
    
                    # a   b  c  d  e  f  g  h  i  j  k  
    R = np.array([  [0,   1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	], # a
                    [0,	0,	0,	0,	0,	1,	1,	1,	0,	0,	0,	], # b
                    [0,	0,	0,	0,	0,	1,	1,	1,	0,	0,	0,	], # c
                    [0,	0,	0,	0,	0,	1,	1,	1,	0,	0,	0,	], # d
                    [0,	0,	0,	0,	0,	1,	1,	1,	0,	0,	0,	], # e
                    [0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	0,	], # f
                    [0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	0,	], # g
                    [0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	0,	], # h
                    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	], # i
                    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	], # j
                    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	]  ]) # k
    
    
    
    # Making a mapping from the state to the locations
    state_to_location = {state: location for location, state in location_to_state.items()} 
    
    # Final function that will return the optimal route
    def route(starting_location, ending_location):
        
        # change the reward for the end location
        R_new = np.copy(R) # make a copy, not a reference
        R_new[location_to_state[ending_location],location_to_state[ending_location]] = 1000
        
        R_new[location_to_state['A'],location_to_state['D']] = -500
        R_new[location_to_state['B'],location_to_state['G']] = 500
        R_new[location_to_state['C'],location_to_state['G']] = 500    
        R_new[location_to_state['D'],location_to_state['G']] = 500    
        R_new[location_to_state['E'],location_to_state['G']] = 500   
             
        #print(R_new)
    
        # Building the AI solution with Q-learning
        # Initializing the Q-Values
        Q = np.array(np.zeros([11,11]))
        # Implementing the Q-Learning process
        for i in range(1000):
            current_state = np.random.randint(0,11)
            playable_actions = []
            
            # loop over the R_new columns
            for j in range(11):
                # if action is playable
                if R_new[current_state, j] > 0:
                    playable_actions.append(j)
                    
            # play a random action from the playable list
            next_state = np.random.choice(playable_actions)
        
            # compute Temporal Difference
            TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
            Q[current_state, next_state] += alpha * TD     
             
        # Navigate trough the Q array
        route = [starting_location] 
        next_location = starting_location
        while (next_location != ending_location):
            starting_state = location_to_state[starting_location]
            next_state = np.argmax(Q[starting_state,])
            next_location = state_to_location[next_state]
            route.append(next_location)
            starting_location = next_location
        return route
    
    
    def best_route(starting_location, intermediary_location, ending_location):
        return route(starting_location, intermediary_location) + route(intermediary_location, ending_location)[1:] # starting from te second location to not get intermediary twice
    
    # Print final route
    print('Route:')
    return json.dumps(route('A','K'))# ''.join(route('A','K'))
    #best_route('E', 'F', 'G')