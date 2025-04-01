'''
Plot the (accuracy, computational cost) vs. iterations obtained from iterative MCTS
'''
import matplotlib.pyplot as plt

# Accuracy obtained from iterative MCTS
iterations = [1, 2, 3, 4, 5, 6]
accuracy = [90.85,91.55,91.78,91.83,91.83,91.83] # from benchmark
cost = [1,1.1,1.2,1.3,1.5,1.7] # estimated


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('MCTS Iterations', fontsize=14) 
ax1.set_ylabel('Final Architecture Accuracy', color=color, fontsize=14) 
ax1.plot(iterations, accuracy, color=color)
ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
ax1.grid(axis='x')

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Relative Cost', color=color, fontsize=14)  ax2.plot(iterations, cost, color=color)
ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
# ax2.grid(True)
ax1.set_xlim(min(iterations), max(iterations))

fig.tight_layout() 
plt.show()
