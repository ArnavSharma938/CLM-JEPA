import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from run_decoder_projected import decision
from train import validate_serialization_endings


@pytest.mark.parametrize('effects,expected', [
    ([.01,.02], 'proceed'), ([0,0], 'stop'), ([-.01,0], 'stop'),
    ([.01,0], 'seed_1301_required'), ([-.01,.02], 'seed_1301_required'),
    ([.01,-.03,.01], 'stop'), ([.01,-.01,.01], 'proceed'),
    ([.04,-.01,-.01], 'stop'),
])
def test_prespecified_decision(effects, expected):
    assert decision(effects) == expected


def test_existing_serialization_validator_uses_collator_labels():
    ids = torch.tensor([[8,3,1,4,3,0], [7,8,3,1,5,3]])
    labels = torch.tensor([[-100,-100,1,4,3,-100], [-100,-100,-100,1,5,3]])
    def collator(rows):
        return dict(input_ids=ids, attention_mask=ids != 0, labels=labels)
    validate_serialization_endings(collator, [{},{}], 3)
    original = labels.clone()
    ids[0,1] = 6
    with pytest.raises(ValueError, match='source truncation'):
        validate_serialization_endings(collator, [{},{}], 3)
    ids[0,1], ids[0,4] = 3, 6
    with pytest.raises(ValueError, match='target truncation'):
        validate_serialization_endings(collator, [{},{}], 3)
    assert torch.equal(original, labels)
