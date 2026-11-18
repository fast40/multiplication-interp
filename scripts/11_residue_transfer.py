"""CPU: after transplants, does the output's residue mod m follow the donor or the original?"""
import numpy as np, json
z = np.load("results/ablate_transplant_outputs.npz"); P = z["P"]; Py = P[z["donor"]]
res = {}
for k in ["transplant_modular", "transplant_magnitude", "transplant_random(mod-size)"]:
    o = z[k]; res[k] = {}
    for m in [2, 3, 4, 5, 7, 8, 9, 10]:
        diff = (P % m) != (Py % m)  # only cases where donor and original disagree mod m
        res[k][m] = dict(follows_donor=float((o % m == Py % m)[diff].mean()), follows_orig=float((o % m == P % m)[diff].mean()),
                         chance=1 / m, n=int(diff.sum()))
    print(k, {m: (round(v["follows_donor"], 2), round(v["follows_orig"], 2)) for m, v in res[k].items()})
json.dump(res, open("results/residue_transfer.json", "w"), indent=1)
