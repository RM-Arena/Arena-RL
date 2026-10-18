import pytest
from arena_rl.bridge import check_provenance


def test_equivalent_engine_sources_allow_platform_specific_binary():
    common={'files':{'a':'same'},'engine_hash':'engine','rules_hash':'rules'}
    check_provenance({**common,'binary_sha256':'mac'},{**common,'binary_sha256':'linux'})


def test_changed_engine_rejected_unless_explicitly_allowed():
    before={'files':{'a':'old'},'engine_hash':'old','rules_hash':'rules','binary_sha256':'a'}
    after={'files':{'a':'new'},'engine_hash':'new','rules_hash':'rules','binary_sha256':'b'}
    with pytest.raises(RuntimeError):check_provenance(before,after)
    check_provenance(before,after,allow=True)
