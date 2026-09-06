"""Protect the oracle's causal indexing and teacher-target shift."""
import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_oracle_transition import rows, inputs
from run_oracle_transition import decoder_weight
from amend_oracle_transition import projection_ceiling
from latent_predictability import TargetBasis, decoder_distribution_metrics


def test_common_product_rows_and_explicit_oracle_only():
    records=[]
    for identity in ("a","b"):
        records.append(dict(reaction_identity=identity,split="train",product_indices=list(range(4,12)),
                            input_ids=torch.arange(15),
                            final_states=torch.arange(8).float()[:,None].expand(8,2048).clone()))
    data=rows(dict(records=records),"train")
    assert len(data["y"])==6
    assert data["token"].tolist()==[8,9,10]*2
    assert data["gold"].tolist()==[9,10,11]*2
    assert data["y"][:,0].tolist()==[4,5,6]*2
    embedding=torch.arange(15).float()[:,None].expand(15,2048)
    a,c,d=(inputs(data,embedding,arm) for arm in ("A","C","D"))
    assert torch.equal(a,c[:,:2048])
    assert torch.equal(a,d[:,6144:8192])
    assert d[0,::2048].tolist()==[0,1,2,3,8]
    assert c[0,::2048].tolist()==[3,8]
    # Changing future hidden states cannot affect any input for the first row.
    records[0]["final_states"][4:]=999
    changed=rows(dict(records=records),"train")
    for arm in ("A","C","D"):
        assert torch.equal(inputs(data,embedding,arm)[0],inputs(changed,embedding,arm)[0])


def test_no_decoder_target_outside_product():
    record=dict(reaction_identity="a",split="test",product_indices=list(range(2,8)),
                input_ids=torch.arange(10),final_states=torch.zeros(6,2048))
    data=rows(dict(records=[record]),"test")
    assert len(data["y"])==1
    assert data["metadata"][0]["decoder_target_index"]==7


def test_projection_ceiling_uses_train_mean_and_equal_reaction_weights():
    basis=TargetBasis(torch.zeros(2),torch.tensor([[1.,0.]]),.9)
    values=torch.tensor([[1.,1.],[2.,0.],[0.,3.]])
    measured=projection_ceiling(values,basis,["a","a","b"])
    assert abs(measured["reaction_mean_r2"]-5/12)<1e-6
    assert abs(measured["reaction_mean_normalized_mse"]-7/12)<1e-6
    assert abs(measured["token_weighted_r2"]-1/3)<1e-6


def test_native_vocabulary_excludes_reserved_argmax_and_probability_mass():
    payload=dict(lm_head=torch.tensor([[1.,0.],[0.,1.],[0.,0.],[100.,100.]]),
                 provenance=dict(native_vocab=3))
    target=torch.tensor([[1.,0.]]);pred=torch.tensor([[0.,1.]])
    full=decoder_weight(payload,False);native=decoder_weight(payload,True)
    f=decoder_distribution_metrics(target@full.T,pred@full.T,torch.tensor([0]))
    n=decoder_distribution_metrics(target@native.T,pred@native.T,torch.tensor([0]))
    assert bool(f["top1_agreement"][0]) and not bool(n["top1_agreement"][0])
    assert n["gold_probability"][0]>f["gold_probability"][0]
