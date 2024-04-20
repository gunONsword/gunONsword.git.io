
import matplotlib.pyplot as plt

# 図の大きさとフォントサイズを設定
plt.rcParams["font.size"] = 20
plt.rcParams['figure.figsize'] = 12, 10


def do_experiment_2species(r_A, r_B, d_A, d_B, s, n0, p0, t, alpha):
    NAt = n0             #r_A=0.8, r_B=0.5, d_A=0.9, d_B=0.1, s=0.1, n0=p0=0.1, α=0.2
    NBt = n0
    Pt = p0
    X = NAt + NBt
    Q = d_A * NAt + d_B * NBt
    datax = []
    datay1 = []
    datay2 = []
    datay3 = []
    datay4 = []

    datax.append(0)
    datay1.append(NAt)
    datay2.append(NBt)
    datay3.append(Pt)
    datay4.append(X)


    for i in range(t - 1):
        NAt = NAt + alpha * NAt * (r_A - X - d_A * Pt)
        NBt = NBt + alpha * NBt * (r_B - X - d_B * Pt)
        Pt = Pt + alpha * Pt * (Q - s)
        X = NAt + NBt
        Q = d_A * NAt + d_B * NBt
        datax.append(i + 1)
        datay1.append(NAt)
        datay2.append(NBt)
        datay3.append(Pt)
        datay4.append(X)

    return (datax, datay1, datay2, datay3, datay4)


dataX, dataY1, dataY2, dataY3, dataY4 = do_experiment_2species(0.8, 0.5, 0.9, 0.1, 0.1, 0.1, 0.1, 2000, 0.2)

plt.plot(dataX, dataY1, dataX, dataY2, dataX, dataY3, dataX, dataY4)
plt.xlabel("t")
plt.ylabel("population size")
plt.legend(["preyA", "preyB", "predator", "preyA+preyB"])
plt.show()
