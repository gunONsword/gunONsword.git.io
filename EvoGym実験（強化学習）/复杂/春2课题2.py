import matplotlib.pyplot as plt
import random as rnd
import matplotlib.animation as animation
import numpy as np

# 设置参数
N = 300  # 总人口数
T = 100  # 时间步数
W = 25   # 网格大小
SEED = 101
PI = 0.1  # 感染概率
PR = 0.05  # 康复概率
PD = 0.01  # 死亡概率
agents = []

## 定义函数
def clip(x):
    if x < 0:
        return x + W
    elif x >= W:
        return x - W
    else:
        return x

## 定义类
class Agent(object):
    def __init__(self, sp):
        self.x = rnd.randint(0, W - 1)
        self.y = rnd.randint(0, W - 1)
        self.s = sp  # 初始状态

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

    def interactions(self):
        if self.s == 'I':
            neighbors_s = [a for a in agents if (abs(a.x - self.x) <= 1 and abs(a.y - self.y) <= 1) and a != self and a.s == 'S']
            for a in neighbors_s:
                if rnd.random() < PI:
                    a.s = 'I'
            if rnd.random() < PR:
                self.s = 'R'
            if rnd.random() < PD:
                self.s = 'D'

# 可视化设置
fig = plt.figure(figsize=[4, 8])
fig.clear()

# 初始化变量
rnd.seed(SEED)
agents = [Agent('S') for i in range(N)]
agents[0].s = 'I'

fs = []
fi = []
fr = []
fd = []

col = ['blue', 'red', 'green', 'black']

# 主循环（动画的回调函数）
def main_loop(t):
    step()
    update(t)

def update(time):
    fig.clear()
    ax1 = fig.add_subplot(2, 1, 1)
    x = [a.x for a in agents]
    y = [a.y for a in agents]
    c = [col.index(a.s) if a.s in col else 0 for a in agents]

    ax1.scatter(x, y, color=[col[i] for i in c])
    ax1.axis([-1, W, -1, W])
    ax1.set_title('t = ' + str(time))

    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(fs, color=col[0], label="S")
    ax3.plot(fi, color=col[1] if 'I' in col else 'purple', label="I")
    ax3.plot(fr, color=col[2], label="R")
    ax3.plot(fd, color=col[3], label="D")

    ax3.set_xlabel('step')
    ax3.legend(loc="upper right")

# 初始化状态可视化
update(0)

# 寻找空间
for a in agents:
    a.findNewSpace()

# 主循环
def step():
    rnd.shuffle(agents)
    for a in agents:
        a.findNewSpace()
        a.interactions()

    fs.append(len([a for a in agents if a.s == 'S']))
    fi.append(len([a for a in agents if a.s == 'I']))
    fr.append(len([a for a in agents if a.s == 'R']))
    fd.append(len([a for a in agents if a.s == 'D']))

ani = animation.FuncAnimation(fig, main_loop, np.arange(0, T), interval=100, repeat=False)
plt.show()
