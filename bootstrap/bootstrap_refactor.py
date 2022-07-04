from __future__ import division, print_function
import numpy as np
from collections import OrderedDict
import pandas as pd
from itertools import product
import argparse

_THRES = 0.8

casa = ["A430_{0}0", "A594_{0}2", "A488_{0}0", "A680_{0}2", "A532_{0}1", "A430_{0}2"]
casb = ["A680_{0}1", "A488_{0}1", "A430_{0}2", "A532_{0}2", "A594_{0}1", "A532_{0}0"]
casc = ["A430_{0}1", "A405_{0}1", "A488_{0}2", "A680_{0}0", "A405_{0}2", "A594_{0}0"]

channels = ["A680", "A594", "A532", "A488", "A430", "A405"]
keys = list(product(channels, ["N", "C"], range(3)))

wdict = np.load("wdict.npz")  # staining wt cells
ddf = pd.read_csv("stain.csv", index_col=0)  # cellular data

round_abs = {}
round_abs[0] = [
    "A430_{0}0",
    "A594_{0}2",
    "A488_{0}0",
    "A680_{0}2",
    "A532_{0}1",
    "A430_{0}2",
]  # round 0 Abs
round_abs[1] = [
    "A680_{0}1",
    "A488_{0}1",
    "A430_{0}2",
    "A532_{0}2",
    "A594_{0}1",
    "A532_{0}0",
]  # round 1 Abs
round_abs[2] = [
    "A430_{0}1",
    "A405_{0}1",
    "A488_{0}2",
    "A680_{0}0",
    "A405_{0}2",
    "A594_{0}0",
]  # round 2 Abs


def compute_percentile(df, wdict):
    dic = OrderedDict()
    keys = list(product(channels, ["N", "C"], range(3)))
    for ch, loc, r in keys:
        key = "{0}_{1}{2}".format(ch, loc, r)
        cells = np.array(df[key])

        ref = wdict[key]
        sortedref = sorted(ref)
        sortedref += [
            np.Inf,
        ]
        sortedref = np.array(sortedref)

        st = []
        for cn in range(cells.shape[0]):
            prc = (np.where(sortedref > cells[cn])[0] / len(ref))[0]
            st.append(prc)
        dic[key] = st
    return pd.DataFrame.from_dict(dic)


def compute_flag_ncratio(ddf, threshold_relative_n=2.0):
    ntdic = {}
    for ch, loc, r in keys:
        key = "{0}_{1}{2}".format(ch, loc, r)
        cells = np.array(ddf[key])

        if loc == "N":
            ccells = np.array(ddf["{0}_{1}{2}".format(ch, "C", r)])
            ntdic[key] = cells > ccells * threshold_relative_n
        else:
            ccells = np.array(ddf["{0}_{1}{2}".format(ch, "N", r)])
            ntdic[key] = cells > ccells * 0
    return ntdic


def gen_random_epitopes():
    cst = [np.random.choice(round_abs[i]).format("C") for i in range(3)] + [
        np.random.choice(round_abs[i]).format("N") for i in range(3)
    ]
    return cst


def calc_acc(csets, percdf, ntdic, num=1000, thres=4.8):

    ini_len = len(csets)
    for i in range(num - ini_len):
        csets.append(gen_random_epitopes())

    counter = 0
    bosu = 0
    for i in range(percdf.shape[0]):
        cscore = []
        for cset in csets:
            scr = np.sum([percdf.iloc[i][cs] if ntdic[cs][i] else 0.5 for cs in cset])
            cscore.append(scr)
        if np.max(cscore) > thres:
            bosu += 1
        else:
            continue
        if np.argsort([-j for j in cscore])[0] < ini_len:
            counter += 1
    return counter / bosu, bosu / percdf.shape[0]


if __name__ == "__main__":

    cset0 = [
        "A594_C2",
        "A488_C1",
        "A594_C0",
        "A532_N1",
        "A532_N0",
        "A430_N1",
    ]  # corresponds to the epitopes expressed in the cell line

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=100)
    parser.add_argument("--nboot", type=int, default=30)
    args = parser.parse_args()

    percdf = compute_percentile(ddf, wdict)
    ntdic = compute_flag_ncratio(ddf)  # Flags for N < 2C
    csets = [
        cset0,
    ]

    st = []
    for _ in range(args.nboot):
        ret = calc_acc(csets[:], percdf, ntdic, args.num, 6.0 * _THRES)
        st.append(ret)
        print(ret)
    pd.DataFrame(st).to_csv(
        "d0cret{0}.csv".format(args.num), header=False, index=False, mode="a"
    )
