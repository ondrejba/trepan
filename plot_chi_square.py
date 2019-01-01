from scipy.stats import chi2
import matplotlib.pyplot as plt
plt.style.use("seaborn-colorblind")


ef1 = 500
ef2 = 500

x = []
y = []

for i in range(1, 999):

    f1 = i
    f2 = 1000 - i

    chi = (((f1 - ef1) ** 2) / (f1 + ef1)) + (((f2 - ef2) ** 2) / (f2 + ef2))

    x.append(i)
    y.append(chi)

threshold = chi2.isf(0.05, 1)

plt.plot(x, y)
plt.plot(x, [threshold for _ in range(len(x))], label="rejection region")

plt.title("chi-square test of an uniform binary set")
plt.xlabel("f1 (f2 = 1000 - f1)")
plt.ylabel("chi-square")

plt.legend()

plt.show()
