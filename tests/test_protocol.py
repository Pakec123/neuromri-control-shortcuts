"""Statistical boundaries matter more than mocking training internals."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binom
from neuromri.common import seed_all
from neuromri.data import task_roles, support_arm, group_scores
from neuromri.statistics import quantile_threshold, temperature, endpoints
from neuromri.verify_data import verify_splits


def test_frozen_group_isolation():
    assert verify_splits("manifests")[0]==25963


def test_paired_positive_groups_equal_negative_budgets():
    c=dict(manifests="manifests",grouping="primary")
    d=task_roles(c,"AD",0,p1=True)
    a=support_arm(d,"AD",0,5101)
    b=support_arm(d,"AD",1,5101)
    assert set(a[a.y==1].group_id)==set(b[b.y==1].group_id)
    assert a[a.y==0].group_id.nunique()==b[b.y==0].group_id.nunique()
    assert not set(a.group_id)&set(d[d.role=="test"].group_id)


def test_binomial_tolerance_coverage_and_unavailable():
    assert np.isnan(quantile_threshold(np.arange(16),tolerance=True))
    for n in [32,64,128]:
        t=quantile_threshold(np.arange(n),tolerance=True)
        r=int(np.floor(t))+1
        assert binom.cdf(r-1,n,.9)>=.95
        assert r==1 or binom.cdf(r-2,n,.9)<.95


def test_group_mean_and_ties():
    d=pd.DataFrame(dict(image_id=[0,1,2],group_id=[1,1,2],y=[0,0,1],normal_cluster=[0,0,-1]))
    g=group_scores(d,[.1,.3,.7])
    np.testing.assert_allclose(g.score,[.2,.7])
    t=quantile_threshold([.1,.1,.2,.2],specificity=.5)
    assert np.mean(np.array([.1,.1,.2,.2])<t)>=.5


def test_positive_temperature_cannot_change_half_threshold():
    from scipy.special import expit, logit
    y=np.array([0,0,1,1])
    p=np.array([.01,.6,.7,.9])
    t=temperature(y,p)
    z=expit(logit(p)/t)
    np.testing.assert_array_equal(p>=.5,z>=.5)
    assert endpoints(y,p)["auroc"]==endpoints(y,z)["auroc"]


def test_paired_bootstrap_does_not_count_repeats_as_people():
    import pytest
    from neuromri.aggregate import paired_bootstrap
    a=pd.DataFrame(dict(group_id=[0,1,2,3],y=[0,0,1,1],score=[.1,.8,.2,.9]))
    b=a.assign(score=[.1,.2,.8,.9])
    r=paired_bootstrap(a,b,draws=50)
    assert r.set_index("metric").loc["auroc","right_minus_left"]==.25
    with pytest.raises(ValueError,match="duplicate groups"):
        paired_bootstrap(pd.concat([a,a]),pd.concat([b,b]),draws=2)


def test_negative_adversary_reverses_only_negative_gradients():
    import torch
    from neuromri.p1_finetune import Reverse
    x=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True)
    mask=torch.tensor([True,False])
    Reverse.apply(x[mask]).sum().backward()
    torch.testing.assert_close(x.grad,torch.tensor([[-1.,-1.],[0.,0.]]))
