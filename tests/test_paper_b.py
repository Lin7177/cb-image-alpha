"""Independent small fixtures for paper-b-v1; no market data is read."""
import numpy as np
import pytest
import torch
from torch import nn

from cb_image_alpha.models.paper_cnn import PaperCNN
from paper_replication.contracts import eligibility, normalize, partition, pixel_statistics
from paper_replication.state import EarlyStop, capture_state, ensemble_probabilities, restore_state


def test_filter_boundaries_and_missing():
    dates = np.array(['2000-11-30','2000-12-01','1999-01-29','1999-01-29','1999-01-29','1999-01-29'],dtype='datetime64[D]')
    keep, counts = eligibility(dates,np.array([0,.1,np.nan,-99,-.1,.1]),np.array([1,1,1,1,0,1]))
    assert keep.tolist()==[True,False,False,False,False,True]
    assert counts=={"raw":6,"invalid_return":2,"invalid_vol_after_return":1,"tail_after_valid":1,"eligible":2}
    with pytest.raises(ValueError,match='development'):
        eligibility(np.array(['2001-01-31']),np.array([.1]),np.array([1]))


def test_fixed_permutation_fixture():
    train,val=partition(10,0)
    assert train.tolist()==[4,1,7,5,3,9,0]
    assert val.tolist()==[8,6,2]
    assert np.array_equal(np.sort(np.concatenate([train,val])),np.arange(10))


def test_train_only_hand_normalization_and_val_perturbation():
    counts=np.array([0,3840,100,200],dtype=np.int64)
    indices=np.array([0,1])
    stats=pixel_statistics(counts,indices)
    assert (stats.mean,stats.std,stats.white_pixels,stats.total_pixels)==(.5,.5,3840,7680)
    counts[2:]=[3840,0]
    assert pixel_statistics(counts,indices)==stats
    images=np.zeros((2,64,60),dtype=np.uint8)
    images[1]=255
    result=normalize(images,stats)
    assert result.dtype==np.float32
    assert np.all(result[0]==-1) and np.all(result[1]==1)
    images[0,0,0]=1
    with pytest.raises(ValueError,match='Nonbinary'):
        normalize(images,stats)
    with pytest.raises(ValueError,match='Degenerate'):
        pixel_statistics(np.zeros(2,dtype=np.int64),indices)


def test_early_stop_and_equal_probability_ensemble():
    early=EarlyStop()
    assert early.observe(0,.7)
    assert not early.observe(1,.7)
    assert early.observe(2,.6)
    assert not early.observe(3,.8)
    assert not early.observe(4,.9) and early.should_stop and early.best_epoch==2
    a=torch.tensor([[0.,0.]])
    b=torch.tensor([[0.,np.log(3)]],dtype=torch.float32)
    result=ensemble_probabilities([a,a,a,a,b])
    assert torch.allclose(result,torch.tensor([[.45,.55]]))
    with pytest.raises(ValueError,match='five'):
        ensemble_probabilities([a])


def test_snapshot_isolation_and_exact_adam_rng_restore(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model=nn.Sequential(nn.Linear(2,3),nn.Dropout(.5),nn.Linear(3,2))
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    generator=torch.Generator().manual_seed(37)
    x=torch.tensor([[1.,2.],[-1.,0.]])
    target=torch.tensor([0,1])
    def step():
        order=torch.randperm(2,generator=generator)
        optimizer.zero_grad()
        loss=nn.functional.cross_entropy(model(x[order]),target[order])
        loss.backward()
        optimizer.step()
    step()
    stopping=EarlyStop(.6,0,1)
    saved=capture_state(model,optimizer,generator,stopping)
    path=tmp_path/'state.pt'
    torch.save(saved,path)
    step()
    expected={k:v.clone() for k,v in model.state_dict().items()}
    restored=restore_state(torch.load(path,weights_only=True),model,optimizer,generator)
    assert restored==stopping
    step()
    assert all(torch.equal(v,expected[k]) for k,v in model.state_dict().items())
    assert any(not torch.equal(v,saved['model'][k]) for k,v in expected.items())


def test_paper_cnn_synthetic_finite_and_save_load(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(2701)
    model=PaperCNN()
    assert sum(p.numel() for p in model.parameters())==708866
    x=torch.zeros(2,1,64,60)
    x[1,:,::2]=1
    out=model(x)
    nn.functional.cross_entropy(out,torch.tensor([0,1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad(): expected=model(x)
    path=tmp_path/'cnn.pt'
    torch.save(model.state_dict(),path)
    loaded=PaperCNN()
    loaded.load_state_dict(torch.load(path,weights_only=True))
    loaded.eval()
    with torch.no_grad(): assert torch.equal(expected,loaded(x))