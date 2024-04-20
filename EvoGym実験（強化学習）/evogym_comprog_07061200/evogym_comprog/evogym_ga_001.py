import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

import matplotlib.animation as animation

import os
import numpy as np
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot
import copy
import imageio

gen=0
ind=0

#https://leetcode.com/problems/number-of-islands/solutions/505712/python-dfs-and-bfs-approach/
class ViableGenotype(object):
    def __init__(self):
        self.zerolist=[]

    def numIslands(self, grid):
        if not grid.any():
            return 0

        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]>=1:
                    self.dfs(grid, i, j)
                    count += 1
        return count

    def dfs(self, grid, i, j):
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or (grid[i][j] < 0):
            return
        elif grid[i][j]==0:
            self.zerolist.append((i, j))
            return
        grid[i][j] = -1
        self.dfs(grid, i+1, j)
        self.dfs(grid, i-1, j)
        self.dfs(grid, i, j+1)
        self.dfs(grid, i, j-1)
    
    def getViable(self, g):
        grid= np.array(g).reshape(W, W)
        ogrid= copy.deepcopy(grid)
        num= self.numIslands(grid)
        zerolistf=self.zerolist
        rnd.shuffle(zerolistf)
        while (num>1):
            zeropos= zerolistf.pop()
            ogrid[zeropos[0]][zeropos[1]]= rnd.randint(1, B-1)
            grid= copy.deepcopy(ogrid)
            num= self.numIslands(grid)
        return(ogrid.flatten())
via= ViableGenotype()

def gtypeToPtype(gtype):
    return(np.array(gtype).reshape(W, W))   

## define classes
class Agent(object):
     
    def __init__(self, gtype):
        self.genotype= gtype[:]
        self.phenotype= None
        self.fitness= 0.0

        self.genotype= via.getViable(self.genotype)

    def getOffspring(self):
        o= Agent(self.genotype)
   
        for i in range(L):
            if (rnd.random()<MUT):
                o.genotype[i]= rnd.randint(0, B-1)
        o.genotype= via.getViable(o.genotype)         

        return(o)
  
    def develop(self, dfunc):
        self.phenotype= dfunc(self.genotype)
  
    def evaluate(self):
        trajectory=[]

        #initialize EvoGym environment
        world = EvoWorld.from_json(os.path.join(ENV))
        world.add_from_array(
            name='robot',
            structure= self.phenotype,
            x=INITPOS[0],
            y=INITPOS[1]
        )

        sim = EvoSim(world)
        sim.reset()
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        frames=[]

        for i in range(T):
            sim.set_action('robot', np.random.uniform(low = BMIN, high = BMAX, size=(sim.get_dim_action_space('robot'),)))
            sim.step()

            #if you use the positional info during the evaluation steps for fitness evaluation, 
            #the code below might be useful.
            #posdata= sim.object_pos_at_time(i, "robot")
            #centerofmass= (np.average(posdata[0]), np.average(posdata[1]))
            #trajectory.append(centerofmass[0])
            if(i%FRAMEINTERVAL==0):
                frames.append(viewer.render('img'))
        
        #fitness definition
        posdata= sim.object_pos_at_time(i, "robot")
        endx= np.average(posdata[0])
        self.fitness= endx if endx>0 else 0
        #print(self.fitness)
        print(f"gen:{gen}_ind:{ind}_fit:{self.fitness:.03f}_gtype:{self.genotype}")
        imageio.mimsave(DIR+f'/{gen}_{ind}_{self.fitness:.03f}_{self.genotype}.gif', frames, format='GIF', fps=FPS) 
 
def selectAnAgentByRoulette(pop):
    total= sum([i.fitness for i in pop])
    val= rnd.random()*total
    for i in pop:
        val-= i.fitness
        if (val<0):
            return(i)

def selectAnAgentByTournament(pop):
    a1= pop[rnd.randint(0, N-1)]
    a2= pop[rnd.randint(0, N-1)]
    if a1.fitness > a2.fitness:
        return(a1)
    else:
        return(a2)
   
def crossover(a1, a2):
    point= rnd.randint(1, L-1)
    for i in range(point, L):
        a1.genotype[i], a2.genotype[i]= a2.genotype[i], a1.genotype[i]
    a1.genotype= via.getViable(a1.genotype)
    a2.genotype= via.getViable(a2.genotype)
  
  
# initialize variables
SEED=101
T= 400
N= 6
G= 20

B= 5
W= 4
L= W*W
BMIN=0.6
BMAX=1.6*2
INITPOS= (3, 1)

ENV= 'my_evironment_flat.json'
DIR= 'experiment_0'

MUT= 0.1
CROSS= 0.15

#exports frames every FRAMEINTERVAL steps
FRAMEINTERVAL= 10
#frame per sec.
FPS= 10

rnd.seed(SEED)
np.random.seed(SEED)

population= [Agent([rnd.randint(0, B-1) for j in range(L)]) for i in range(N)]

if not os.path.exists(DIR):
    os.makedirs(DIR)

#some variables for graphs
averageFitness= []
bestFitness= []
best= population[0]

# events in a step
def step():
    global population, pp, pf, gen, ind, best
    #fitness evaluation
    best= population[0]
    for a in population:
        a.develop(gtypeToPtype)
        a.evaluate()
        if(a.fitness>best.fitness):
            best= a
        ind+= 1
    averageFitness.append(np.average([a.fitness for a in population]))
    bestFitness.append(best.fitness)
 
    print(f"gen:{gen}_best:{best.fitness:0.3f}_{best.genotype}")
    #evolution
    newpop= []
    for i in range(int(N/2)):
        n1= selectAnAgentByRoulette(population).getOffspring()
        n2= selectAnAgentByRoulette(population).getOffspring()
#        n1= selectAnAgentByTournament(population).getOffspring()
#        n2= selectAnAgentByTournament(population).getOffspring()        
 
        if rnd.random()<CROSS:
            crossover(n1, n2)
        newpop.append(n1)
        newpop.append(n2)
    
    population= newpop

f= open(DIR+"/result.csv", "w")
for i in range(G):
    gen= i
    ind= 0
    step()
    f.write(f"{averageFitness[-1]}\t{bestFitness[-1]}\t{best.genotype}\n")
f.close()


fig= plt.figure()
ax1= fig.add_subplot(2, 1, 1)
ax1.plot(averageFitness)
ax1.plot(bestFitness)
ax1.set_xlabel("generaiton")
ax1.set_ylabel("average / best fitness")    
        
fig.tight_layout()
plt.savefig(DIR+"/result.pdf")



