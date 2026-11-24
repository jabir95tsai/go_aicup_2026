import argparse, sys
import scripts.strength.rank_opp_llr as rl
rl.MUS = (2.0,)
ns = argparse.Namespace(job=sys.argv[1], force=False, llr="llr_g40.npz", alphas="0.45,0.55,0.65", tag="opp7-l10",
                        local_cells=None, experiment=None)
rl.fuse(ns)
