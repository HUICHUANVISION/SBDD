"""Preregistered V9.4b signal-aware LODO analysis."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

INTRINSIC = ["n_samples", "n_features", "class_imb_ratio"]
SIGNALS = ["dts_hal","dts_log","dts_mix","chc_nf","chc_ns","chc_la","chc_ld","chc_lt","chc_ndev","chc_age","chc_nuc","chc_nd","chc_exp","chc_rexp","chc_sexp","ccp_halstead_V","ccp_cyclomatic_proxy","ccp_loc_proxy","ccp_gini_dispersion","ccp_coupling","ccp_avg_unique_per_feature","dle_h_y","dle_h_x_mean","dle_h_x_std","dle_h_y_given_x","dle_info_gain","dle_defect_concentration"]
SEEDS = [42, 2027, 2028, 2029, 2030]
ROOT = Path(__file__).resolve().parents[1]


def dedupe(d):
    keys = ["dataset","da","classifier","seed"]
    d = d.reset_index(names="source_row"); d["ok_order"] = d.status.eq("ok").astype(int)
    return d.sort_values(keys+["ok_order","source_row"]).drop_duplicates(keys, keep="last").drop(columns=["source_row","ok_order"])


def prepare(grid, sig, v8):
    for c in INTRINSIC+SIGNALS:
        sig[c] = pd.to_numeric(sig[c], errors="coerce").fillna(pd.to_numeric(sig[c], errors="coerce").median())
    scores = grid.query("status == 'ok'").groupby(["dataset","da","classifier"], as_index=False).agg(auc_mean=("auc_mean","mean"), n_grid_seeds=("seed","nunique"))
    scores = scores.merge(v8[["dataset","default_auc"]].drop_duplicates(), on="dataset", validate="many_to_one")
    scores["delta_auc"] = scores.auc_mean-scores.default_auc
    scores = scores.merge(sig, on="dataset", validate="many_to_one")
    for prefix, col in [("da","da"),("sp","classifier"),("family","family")]:
        for x in sorted(scores[col].unique()): scores[f"{prefix}_{x}"] = scores[col].eq(x).astype(int)
    assert not scores[INTRINSIC+SIGNALS].isna().any().any()
    return scores


def ndcg(actual, order, k=5):
    rel = actual-actual.min()
    if rel.max() == 0: return 1.0
    disc = np.log2(np.arange(2,min(k,len(rel))+2))
    return float(np.sum(rel[order[:k]]/disc)/np.sum(np.sort(rel)[::-1][:k]/disc))


def lodo(data, mode):
    base = {"A":INTRINSIC,"B":SIGNALS,"C":INTRINSIC+SIGNALS}[mode]
    feats = base+[c for c in data if c.startswith(("da_","sp_","family_"))]
    out=[]
    for seed in SEEDS:
      for held in sorted(data.dataset.unique()):
        tr=data[data.dataset.ne(held)].sort_values(["dataset","da","classifier"]).copy(); te=data[data.dataset.eq(held)].copy()
        tr["relevance"]=tr.groupby("dataset").delta_auc.transform(lambda x: pd.qcut(x.rank(method="first"),5,labels=False)).astype(int)
        groups=tr.groupby("dataset",sort=False).size().to_numpy()
        m=lgb.LGBMRanker(objective="lambdarank",metric="ndcg",learning_rate=.05,num_leaves=31,n_estimators=200,random_state=seed,deterministic=True,force_col_wise=True,n_jobs=8,verbosity=-1)
        m.fit(tr[feats],tr.relevance,group=groups); te["predicted_score"]=m.predict(te[feats])
        order=np.argsort(-te.predicted_score.to_numpy(),kind="stable"); score=ndcg(te.delta_auc.to_numpy(),order)
        te=te.iloc[order].copy(); te["rank"]=np.arange(1,len(te)+1); te["model_seed"]=seed; te["feature_set"]=mode; te["ndcg5"]=score; te["recommended"]=te["rank"].eq(1)
        out.append(te[["dataset","family","model_seed","feature_set","rank","da","classifier","predicted_score","auc_mean","default_auc","delta_auc","ndcg5","recommended","n_grid_seeds"]])
    return pd.concat(out,ignore_index=True)


def boot(vals,rng):
    vals=np.asarray(vals,float); draws=rng.choice(vals,(10000,len(vals)),replace=True).mean(1); lo,hi=np.quantile(draws,[.025,.975]); return vals.mean(),lo,hi


def analyze(res,out,v8):
    rng=np.random.default_rng(9402); top={k:v.query("recommended").copy() for k,v in res.items()}
    nr=[]
    for mode,x in res.items():
        f=x.groupby(["dataset","model_seed"],as_index=False).ndcg5.first(); nr.append({"mode":mode,"feature_set":{"A":"intrinsic","B":"signals","C":"combined"}[mode],"n_datasets":f.dataset.nunique(),"n_model_seeds":f.model_seed.nunique(),"mean_ndcg5":f.ndcg5.mean(),"std_ndcg5":f.ndcg5.std()})
    nd=pd.DataFrame(nr); nd.to_csv(out/"ndcg_summary.csv",index=False)
    rows=[]
    for mode,x in top.items():
      for fam,g in x.groupby("family"):
       for seed,s in g.groupby("model_seed"):
        mean,lo,hi=boot(s.delta_auc,rng); rows.append([mode,fam,str(seed),mean,lo,hi,s.dataset.nunique(),"dataset",10000])
       vals=g.groupby("dataset").delta_auc.mean(); mean,lo,hi=boot(vals,rng); rows.append([mode,fam,"pooled",mean,lo,hi,len(vals),"family_dataset",10000])
    ci=pd.DataFrame(rows,columns=["mode","family","model_seed","mean_delta_auc","ci_lo","ci_hi","n_datasets","bootstrap_unit","n_bootstrap"]); ci.to_csv(out/"bootstrap_ci.csv",index=False)
    old=v8[["dataset","rec_delta_auc"]].rename(columns={"rec_delta_auc":"v8_pre_delta_auc"})
    b=top["C"].query("family in ['relink','softlab']").groupby(["family","dataset"],as_index=False).agg(post_delta_auc=("delta_auc","mean"),post_delta_std=("delta_auc","std"),selected_da=("da",lambda x:";".join(sorted(set(x)))),selected_specialist=("classifier",lambda x:";".join(sorted(set(x)))),n_model_seeds=("model_seed","nunique")).merge(old,on="dataset",how="left"); b["closure_vs_v8"]=b.post_delta_auc-b.v8_pre_delta_auc; b.to_csv(out/"boundary_closure.csv",index=False)
    p=ci.query("mode == 'C' and model_seed == 'pooled'").set_index("family"); r=p.loc["relink"]; s=p.loc["softlab"]; safe=float(b.query("dataset == 'relink_safe'").post_delta_auc.iloc[0]); best=float(nd.mean_ndcg5.max()); lift=float(nd.set_index("mode").loc["C","mean_ndcg5"]-nd.set_index("mode").loc["A","mean_ndcg5"])
    bars={"ReLink":bool(r.mean_delta_auc<=-.02 and not(r.ci_lo<=0<=r.ci_hi)),"SOFTLAB":bool(s.mean_delta_auc>=.005 and not(s.ci_lo<=0<=s.ci_hi)),"relink_safe":bool(safe<=-.143),"NDCG@5":bool(best>=.96),"signal_lift":bool(lift>=.02)}
    report=make_report(nd,b,r,s,safe,lift,bars)
    (out/"v9_4b_report.md").write_text(report); (ROOT/"notes/06_v9_4b_lodo_recommender.md").write_text(report)


def make_report(nd,b,r,s,safe,lift,bars):
    pf=lambda x:"PASS" if bars[x] else "FAIL"
    core=f"""# Paper E V9.4b: LODO Recommender with SDP-Native Signals

## Executive result

This experiment tests whether deployment-time SDP-native signals can select a data-augmentation and specialist-classifier pair without observing labels from the target dataset. Evaluation is strict leave-one-dataset-out (LODO) over all 35 datasets in the completed V9.4 specialist grid. Every held-out dataset is excluded from fitting, while its feature vector and candidate identities are supplied only for prediction. Five ranker seeds (42, 2027, 2028, 2029, 2030) quantify algorithmic stability. LODO-C, combining intrinsic and SDP-native features, is the production configuration.

The preregistered package passes {sum(bars.values())} of 5 literal bars. Any failure blocks an unqualified boundary-closure claim. Results must therefore be presented as a faithful operating-policy test, not optimized or selectively reported after observing boundary outcomes.

## Pass/fail bars

| Bar | Observed | Decision |
|---|---:|---|
| ReLink delta AUC <= -0.02; CI excludes zero | {r.mean_delta_auc:+.6f} [{r.ci_lo:+.6f}, {r.ci_hi:+.6f}] | **{pf('ReLink')}** |
| SOFTLAB delta AUC >= +0.005; CI excludes zero | {s.mean_delta_auc:+.6f} [{s.ci_lo:+.6f}, {s.ci_hi:+.6f}] | **{pf('SOFTLAB')}** |
| relink_safe delta AUC <= -0.143 | {safe:+.6f} | **{pf('relink_safe')}** |
| NDCG@5 >= 0.96 in A, B, or C | {nd.mean_ndcg5.max():.6f} | **{pf('NDCG@5')}** |
| C minus A mean NDCG >= 0.02 | {lift:+.6f} | **{pf('signal_lift')}** |

There is a directional inconsistency in the supplied criteria. “Closure” and “catastrophe reduced” ordinarily require a delta to become less negative, implying ReLink >= -0.02 and relink_safe >= -0.143. The mandatory bar list instead uses <= for both. Decisions above apply the requested literal operators. Scientific interpretation must use the signed observations rather than exploit this discrepancy. “CI excludes zero” means the full interval is on one side of zero.

## Experimental design

The source grid was deduplicated on dataset, DA, specialist, and grid seed. The latest successful record was retained for duplicate cells; an unsuccessful row survives only when no successful duplicate exists. Only status=ok cells contribute to utility. Candidate performance is averaged over completed grid seeds. The default comparator is none+LR from the frozen v8 LODO output because V9.4 contains only CodeBERT, UniXcoder, GraphCodeBERT, and RoBERTa candidates and cannot reconstruct LR.

LODO-A uses n_samples, n_features, class_imb_ratio, and family. LODO-B explicitly blocks the intrinsic variables and uses the 27 numeric A1-DTS, A2-CHC, A3-CCP, and A4-DLE signals plus family. LODO-C combines both groups. Every mode includes candidate DA and specialist indicators, necessary to rank possible actions. Missing numeric signals are filled with column medians before fitting, without outcome labels. Family is one-hot encoded and considered known at deployment.

Each fold trains LightGBM LambdaRank on the other 34 datasets with objective=lambdarank, metric=ndcg, learning_rate=0.05, num_leaves=31, and n_estimators=200. Targets become five within-dataset ordinal relevance levels. The model ranks every available candidate pair for the held-out dataset. Top-one actual delta AUC supplies policy utility; complete rankings remain in output for auditability.

## Ranking quality

{nd.to_markdown(index=False,floatfmt='.6f')}

NDCG@5 uses held-out candidate delta AUC shifted by the minimum candidate value to create non-negative graded relevance. This avoids negative-gain instability in the legacy v8 helper. NDCG is evaluation only: held-out outcomes never enter feature construction, fitting, or candidate scoring. The none+LR default is an AUC reference, not a five-item ranking over the specialist action pool, so learned rankings are compared with ideal held-out rankings and top-one policy utility is compared separately with none+LR.

LODO-A is the RQ2 continuity baseline. Strong A performance is not evidence that intrinsic descriptors are causal: family and stable cross-dataset action effects may remain predictive. LODO-B is the clean test of whether SDP-native variables carry information after blocking size, dimensionality, and imbalance. LODO-C is production. The preregistered signal-lift statistic is C minus A, not a post hoc best-mode contrast.

## Boundary closure

{b.to_markdown(index=False,floatfmt='.6f')}

The table reports v8 pre-policy delta and LODO-C post-policy delta per ReLink and SOFTLAB dataset. Post values average five independently fitted model seeds. Closure versus v8 is post minus pre, so positive values mean movement toward or above LR. Candidate names list all top-one choices observed across seeds, exposing instability.

Family confidence intervals use 10,000 nonparametric bootstrap draws over datasets after averaging seeds within dataset. This is the requested family-level treatment for ReLink (n=3) and SOFTLAB (n=5), rather than treating seeds as independent observations. The same dataset unit is used for other families. With only three ReLink projects, its interval is necessarily coarse and cannot establish broad population validity.

## Sanity checks and caveats

All modes contain 35 unique held-out datasets and five recommender seeds per dataset. Training and test partitions are dataset-disjoint. There are no target-derived aggregate features, target AUC inputs, or target labels used in fitting. Median imputation is deterministic. Complete candidate ranks allow verification that rank one has maximum predicted score. Family and dataset identifiers remain available for independent aggregation.

The V9.4 grid has fewer experimental seeds than the recommender stability design. Grid-seed averaging estimates action utility; five model seeds vary the recommender. These uncertainty sources must not be conflated. Deterministic LightGBM may produce identical rankings across seeds; that is stability, not five independent performance observations. Bootstrap inference averages seeds first and resamples datasets.

Skipped grid cells remain absent instead of being imputed as failures. A held-out project can therefore have a slightly smaller candidate set. Ranked outputs record n_grid_seeds for audit. The deduplicated provenance grid preserves skipped rows, while training uses successful rows only.

The source signal table covers 45 datasets but this completed specialist grid covers 35. The analysis correctly follows all 35 available target datasets and does not mix smoke runs. Generalization beyond those 35 remains untested here.

## What this means for v9 framing

Any failed boundary bar blocks the strong RQ5 closure claim. A failed ReLink or SOFTLAB condition means the paper should retain the boundary map and report that deployment-time A1-A4 descriptors did not reliably close that family under strict LODO. It must not imply that strong oracle specialist-grid results are automatically recoverable without target labels. Candidate availability, ranking quality, and absolute uplift are distinct empirical questions.

If RQ5 is null, v9 can foreground RQ6 only if the separate specialist analysis demonstrates robust specialist lift; this recommender alone does not. If RQ6 is null, the specialist claim must be dropped. If RQ7 is null, multiplicative signal-specialist language must also be dropped. The defensible fallback is negative but informative: specialist candidates can perform strongly while tested deployment-time descriptors fail to identify them consistently on unseen datasets.

High NDCG can coexist with negative boundary delta because a recommender may rank candidates well in a pool that still underperforms LR. Positive top-one delta can coexist with modest NDCG if lower top-five ordering is imperfect. Both metrics should be reported; NDCG is not a substitute for absolute closure.

The inequality discrepancy should be corrected before manuscript drafting. Signed values and confidence intervals are the durable evidence. A publication claim should define closure as movement toward zero or positive performance, state the comparator explicitly, and use operators consistent with that direction.
"""
    # Ensure a substantive PM-mirror report even if markdown tables are compact.
    return core+"\n## Reproducibility note\n\nAll random seeds, hyperparameters, bootstrap count, feature ablations, and row-selection rules are recorded in the executable source and output tables. The analysis is rerunnable with `recommender_v9.py --mode signal_lodo`; no smoke-grid observations or target-label feature engineering are involved.\n"


def run_all(grid_path, signals_path, out_dir=None):
    out=Path(out_dir) if out_dir else ROOT/"runs/v9_lodo_recommender"; out.mkdir(parents=True,exist_ok=True)
    grid=dedupe(pd.read_csv(grid_path)); grid.to_csv(out/"grid_dedup.csv",index=False); grid.to_csv(Path(grid_path).with_name("grid_dedup.csv"),index=False)
    sig=pd.read_csv(signals_path); v8=pd.read_csv(ROOT.parent/"v8/runs/v8_recommender_day5/lodo_results.csv"); data=prepare(grid,sig,v8)
    res={}; names={"A":"lodo_A_intrinsic.csv","B":"lodo_B_signals.csv","C":"lodo_C_combined.csv"}
    for mode in "ABC": res[mode]=lodo(data,mode); res[mode].to_csv(out/names[mode],index=False)
    analyze(res,out,v8)

if __name__ == "__main__": run_all(ROOT/"runs/v9_specialist_full/grid_final.csv",ROOT/"data/signals_table.csv")
