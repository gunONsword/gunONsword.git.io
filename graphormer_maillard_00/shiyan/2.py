import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import matplotlib.animation as animation
# Set parameters
N = 500  # Population size
T = 100  # Number of time steps
W = 30   # Grid size
SEED = 101
agents = []
infection_probability = 0.4  # Probability of infection
recovery_probability = 0.3  # Probability of recovery

colorlist = ['blue', 'red', 'green']  # Colors for visualization

# Define classes
class Agent(object):
    def __init__(self, state):
        self.x = rnd.randint(0, W - 1)
        self.y = rnd.randint(0, W - 1)
        self.state = state  # Initial state (S for susceptible, I for infected, R for recovered)

    def randomwalk(self):
        self.x += rnd.randint(-1, 1)
        self.y += rnd.randint(-1, 1)
        self.x = clip(self.x)
        self.y = clip(self.y)

    def isOverlapped(self):
        for a in agents:
            if (a.x == self.x and a.y == self.y) and (a != self):
                return True
        return False

    def findNewSpace(self):
        self.randomwalk()
        if self.isOverlapped():
            self.findNewSpace()

    def infect(self):
        self.state = 'I'  # Change state to infected

    def recover(self):
        self.state = 'R'  # Change state to recovered

    def update(self):
        if self.state == 'I':  # If agent is infected
            if rnd.random() < recovery_probability:
                self.recover()
        else:  # If agent is susceptible
            neighbors = [a for a in agents if abs(a.x - self.x) <= 1 and abs(a.y - self.y) <= 1 and a != self]
            for neighbor in neighbors:
                if neighbor.state == 'I' and rnd.random() < infection_probability:
                    self.infect()

def clip(x):
    if x < 0:
        return (x + W)
    elif x >= W:
        return (x - W)
    else:
        return (x)

def update(t):
    fig.clear()
    ax1 = fig.add_subplot(2, 2, 1)
    x = [a.x for a in agents]
    y = [a.y for a in agents]
    c = [colorlist[a.p] for a in agents]
    ax1.scatter(x, y, color=c)
    ax1.axis([-1, W, -1, W])
    ax1.set_title('t = ' + str(t))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # ax2 = fig.add_subplot(2, 2, 3)
    # s = [a.s for a in agents]
    # ax2.hist(s, 10)
    # ax2.set_xlabel('satisfaction')
    # ax2.set_ylabel('frequency')
    # ax3 = fig.add_subplot(2, 2, 4)
    # ax3.plot(averageSatisfaction)
    # ax3.set_xlabel('t')
    # ax3.set_ylabel('averageSatisfaction')

    plt.tight_layout()
# Initialize variables
rnd.seed(SEED)

agents = [Agent('S' if i % 2 == 0 else 'I') for i in range(N)]
for agent in agents:
    agent.findNewSpace()

# Main loop
susceptible_counts = []
infected_counts = []
recovered_counts = []

for t in range(T):
    susceptible_count = len([a for a in agents if a.state == 'S'])
    infected_count = len([a for a in agents if a.state == 'I'])
    recovered_count = len([a for a in agents if a.state == 'R'])

    susceptible_counts.append(susceptible_count)
    infected_counts.append(infected_count)
    recovered_counts.append(recovered_count)

    rnd.shuffle(agents)
    for a in agents:
        a.update()

# Plotting the results
time = np.arange(T)
plt.plot(time, susceptible_counts, label='Susceptible')
plt.plot(time, infected_counts, label='Infected')
plt.plot(time, recovered_counts, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Agents')
plt.legend()
plt.show()



ani = animation.FuncAnimation(fig, main_loop, np.arange(0, T), interval=25, repeat=False)
plt.show()