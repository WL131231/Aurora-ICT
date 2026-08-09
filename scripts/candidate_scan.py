import sys,os,time; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bt_par import _load_full,_resample,cached_setup_timeline
from aurora_ict.backtest.replay import BacktestConfig,run_backtest_from_timeline
BASE=dict(htf_ema_bias="align",htf_align_threshold=2,sl_liq_cap=True,min_confluence=5,
    sl_dist_mult=4.0,setup_stale_bars=3,apply_cisd=True,apply_po3=True,disable_time_filter=False,
    size_pct=0.9,ote_level=0.707,min_rr=2.0,tp_rr_override=0.0,entry_ttl_bars=6,
    trail_trigger=2.0,trail_dist=1.5,partial_tp_rr=1.5,partial_be=True)
# 캐시된 것 먼저 → 빠른 결과, TRX/WLD 는 빌드
SYMS=["FILUSDT","BCHUSDT","ENAUSDT","NEARUSDT","ARBUSDT","BTCUSDT","SOLUSDT","TRXUSDT","WLDUSDT"]
def gk(bt,df5):
    mags=[abs(t.entry_trend_pct) for t in bt.trades if not(17<=df5.index[t.entry_idx].hour<21)]
    q70=np.percentile(mags,70) if mags else 0
    o=[]
    for t in bt.trades:
        h=df5.index[t.entry_idx].hour
        if 17<=h<21: continue
        sg=1.0 if t.direction=="long" else -1.0
        if abs(t.entry_trend_pct)<q70 and t.entry_trend_pct*sg<0: continue
        o.append(t)
    return o
OUT="candidate_scan_result.txt"
open(OUT,"w").write(f"{'페어':<10}{'거래':>6}{'net%':>9}{'승률':>7}{'RR':>6}{'빈도/일':>8}{'H1':>8}{'H2':>8}\n")
for sym in SYMS:
    t0=time.time()
    try:
        df5=_resample(_load_full(sym)); cfg=BacktestConfig(**BASE)
        tl=cached_setup_timeline(df5,cfg,sym); bt=run_backtest_from_timeline(df5,tl,cfg)
        k=gk(bt,df5); n=len(k); net=sum(t.net_pnl_pct for t in k)
        w=sum(1 for t in k if t.net_pnl_pct>0)
        wins=[t.net_pnl_pct for t in k if t.net_pnl_pct>0]; los=[t.net_pnl_pct for t in k if t.net_pnl_pct<0]
        rr=(sum(wins)/len(wins))/abs(sum(los)/len(los)) if wins and los else 0
        span=max((df5.index[-1]-df5.index[0]).days,1)
        mid=df5.index[len(df5)//2].value
        h1=sum(t.net_pnl_pct for t in k if df5.index[t.entry_idx].value<mid)
        h2=sum(t.net_pnl_pct for t in k if df5.index[t.entry_idx].value>=mid)
        line=f"{sym:<10}{n:>6}{net:>+9.1f}{100*w/n if n else 0:>6.0f}%{rr:>6.2f}{n/span:>8.3f}{h1:>+8.1f}{h2:>+8.1f}"
    except Exception as e:
        line=f"{sym:<10} ERR {str(e)[:40]}"
    open(OUT,"a").write(line+f"  [{time.time()-t0:.0f}s]\n")
    print(line,flush=True)
print("DONE",flush=True)
