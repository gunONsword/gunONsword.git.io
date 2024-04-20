# 开发时间:1/5/2023 下午 10:19|
import matplotlib.pyplot as plt

# 図の大きさとフォントサイズを設定
plt.rcParams["font.size"] = 20
plt.rcParams['figure.figsize'] = 12, 10

def do_logistic_growth(k, r, n0, t):
    nt = n0
    datax = []
    datay = []
    datar = []

    for i in range(0,t):
        nt = nt + r * nt * (1.0 - nt / k)

        if i >= 251 and i <= 300:
            datax.append(i )
            datay.append(nt)
            datar.append(r)

    return (datax, datay, datar)

for rn in range(100,300):
    dataX, dataY, dataR = do_logistic_growth(100.0, float(rn/100), 1.0, 300)
    plt.plot(dataR, dataY, '.')

plt.xlabel("r")
plt.ylabel("Nt")
plt.show()
