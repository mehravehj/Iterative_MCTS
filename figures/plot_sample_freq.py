'''
plot sampling frequency of top-5 architectures fro uniform, Boltzmann, MCTS_default, MCTS_ours.
First iteration is warm-up/unifrom sampling. 
'''

import matplotlib.pyplot as plt

iterations = [1, 2, 3, 4, 5, 6] 
uniform = [1,1,1,1,1,1] # basis for normalization
Boltzmann = [1,0.72,0.51,0.23,0.01, 0.01]
MCTS_default = [1,0.91,0.67,0.51,0.32, 0.33] 
MCTS_dist = [1,1.08,1.24,1.31,1.50, 1.75] 
MCTS_ours = [1,0.74,0.85,1.2,1.64, 1.60] 


plt.figure(figsize=(10, 6))

plt.plot(iterations, uniform, marker='o', linestyle='-', label='Uniform')
plt.plot(iterations, Boltzmann, marker='s', linestyle='-', label='Boltzmann')
plt.plot(iterations, MCTS_default, marker='^', linestyle='-', label='MCTS-default')
plt.plot(iterations, MCTS_ours, marker='*', linestyle='-', label='MCTS-ours')

plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Relative Sampling Frequency', fontsize=14)
plt.grid(True)
plt.legend()
plt.xlim(min(iterations), max(iterations)) # Ensure x-axis limits match the data
plt.legend(fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()
