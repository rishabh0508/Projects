from apyori import apriori
import pandas as pd
datasets = pd.read_csv("Market_Basket_Optimisation.csv", header=None).values
ls = []
print(len(datasets[]))
for i in range(0, 7501):
    ls.append([(str(datasets[i, j])) for j in range(0, 20)])

model = apriori(transactions=ls, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
pred = list(model)
print(pred)

# lhs,rhs = [],[]
#
# conf = []
# sup = []
# lift = []
#
# for i in range(len(pred)):
#     lhs.append(tuple(pred[i][2][0][0])[0])
#     rhs.append(tuple(pred[i][2][0][1])[0])
#     conf.append(pred[i][2][0][2])
#     lift.append(pred[i][2][0][3])
#     sup.append(pred[i][1])
# x = zip(lhs,rhs,sup,conf,lift)
# res = pd.DataFrame(x, columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
# print(res)
# print("----------------------------------------------------------")
# r = res.nlargest(n=10, columns='Lift')
# print(r)
