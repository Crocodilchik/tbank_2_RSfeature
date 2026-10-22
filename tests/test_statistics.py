import polars as pl
from Comparison.report import noninferiority

def test_noninferiority_accounts_for_uncertain_baseline():
    a=pl.DataFrame({'user':[1,2,3],'ndcg10':[.1,.2,.5]})
    better=pl.DataFrame({'user':[1,2,3],'ndcg10':[.099,.198,.495]})
    worse=pl.DataFrame({'user':[1,2,3],'ndcg10':[.097,.194,.485]})
    assert noninferiority(a,better)['ci_low']>0
    assert noninferiority(a,worse)['ci_high']<0

def test_user_order_does_not_change_bootstrap():
    a=pl.DataFrame({'user':[1,2,3],'ndcg10':[.1,.2,.5]})
    b=pl.DataFrame({'user':[1,2,3],'ndcg10':[.099,.198,.495]})
    assert noninferiority(a,b)==noninferiority(a.reverse(),b.reverse())
