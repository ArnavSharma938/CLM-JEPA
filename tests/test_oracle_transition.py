"""Protect the oracle's causal indexing and teacher-target shift."""
import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_oracle_transition import rows, inputs


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
