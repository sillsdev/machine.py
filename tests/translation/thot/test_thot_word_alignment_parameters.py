from machine.translation.thot import ThotWordAlignmentModelType, ThotWordAlignmentParameters


def test_get_fast_align_iteration_count_default() -> None:
    parameters = ThotWordAlignmentParameters()
    assert parameters.get_fast_align_iteration_count(ThotWordAlignmentModelType.FAST_ALIGN) == 4
    assert parameters.get_fast_align_iteration_count(ThotWordAlignmentModelType.IBM1) == 0


def test_get_fast_align_iteration_count_set() -> None:
    parameters = ThotWordAlignmentParameters(fast_align_iteration_count=2)
    assert parameters.get_fast_align_iteration_count(ThotWordAlignmentModelType.FAST_ALIGN) == 2
    assert parameters.get_fast_align_iteration_count(ThotWordAlignmentModelType.IBM1) == 0


def test_get_ibm1_iteration_count_default() -> None:
    parameters = ThotWordAlignmentParameters()
    assert parameters.get_ibm1_iteration_count(ThotWordAlignmentModelType.IBM1) == 4
    assert parameters.get_ibm1_iteration_count(ThotWordAlignmentModelType.IBM4) == 5
    assert parameters.get_ibm1_iteration_count(ThotWordAlignmentModelType.FAST_ALIGN) == 0


def test_get_ibm1_iteration_count_set() -> None:
    parameters = ThotWordAlignmentParameters(ibm1_iteration_count=2)
    assert parameters.get_ibm1_iteration_count(ThotWordAlignmentModelType.IBM1) == 2
    assert parameters.get_ibm1_iteration_count(ThotWordAlignmentModelType.IBM4) == 2
    assert parameters.get_ibm1_iteration_count(ThotWordAlignmentModelType.FAST_ALIGN) == 0


def test_get_ibm2_iteration_count_default() -> None:
    parameters = ThotWordAlignmentParameters()
    assert parameters.get_ibm2_iteration_count(ThotWordAlignmentModelType.IBM2) == 4
    assert parameters.get_ibm2_iteration_count(ThotWordAlignmentModelType.IBM4) == 0
    assert parameters.get_ibm2_iteration_count(ThotWordAlignmentModelType.FAST_ALIGN) == 0


def test_get_ibm2_iteration_count_set() -> None:
    parameters = ThotWordAlignmentParameters(ibm2_iteration_count=2)
    assert parameters.get_ibm2_iteration_count(ThotWordAlignmentModelType.IBM2) == 2
    assert parameters.get_ibm2_iteration_count(ThotWordAlignmentModelType.IBM4) == 2
    assert parameters.get_ibm2_iteration_count(ThotWordAlignmentModelType.FAST_ALIGN) == 0


def test_get_hmm_iteration_count_default() -> None:
    parameters = ThotWordAlignmentParameters()
    assert parameters.get_hmm_iteration_count(ThotWordAlignmentModelType.HMM) == 4
    assert parameters.get_hmm_iteration_count(ThotWordAlignmentModelType.IBM4) == 5
    assert parameters.get_hmm_iteration_count(ThotWordAlignmentModelType.FAST_ALIGN) == 0


def test_get_hmm_iteration_count_set() -> None:
    parameters = ThotWordAlignmentParameters(hmm_iteration_count=2)
    assert parameters.get_hmm_iteration_count(ThotWordAlignmentModelType.HMM) == 2
    assert parameters.get_hmm_iteration_count(ThotWordAlignmentModelType.IBM4) == 2
    assert parameters.get_hmm_iteration_count(ThotWordAlignmentModelType.FAST_ALIGN) == 0


def test_get_hmm_iteration_count_ibm2_set() -> None:
    parameters = ThotWordAlignmentParameters(ibm2_iteration_count=2)
    assert parameters.get_hmm_iteration_count(ThotWordAlignmentModelType.HMM) == 4
    assert parameters.get_hmm_iteration_count(ThotWordAlignmentModelType.IBM4) == 0
    assert parameters.get_hmm_iteration_count(ThotWordAlignmentModelType.FAST_ALIGN) == 0


def test_get_ibm3_iteration_count_default() -> None:
    parameters = ThotWordAlignmentParameters()
    assert parameters.get_ibm3_iteration_count(ThotWordAlignmentModelType.IBM3) == 4
    assert parameters.get_ibm3_iteration_count(ThotWordAlignmentModelType.IBM4) == 5
    assert parameters.get_ibm3_iteration_count(ThotWordAlignmentModelType.FAST_ALIGN) == 0


def test_get_ibm3_iteration_count_set() -> None:
    parameters = ThotWordAlignmentParameters(ibm3_iteration_count=2)
    assert parameters.get_ibm3_iteration_count(ThotWordAlignmentModelType.IBM3) == 2
    assert parameters.get_ibm3_iteration_count(ThotWordAlignmentModelType.IBM4) == 2
    assert parameters.get_ibm3_iteration_count(ThotWordAlignmentModelType.FAST_ALIGN) == 0


def test_get_ibm4_iteration_count_default() -> None:
    parameters = ThotWordAlignmentParameters()
    assert parameters.get_ibm4_iteration_count(ThotWordAlignmentModelType.IBM4) == 4
    assert parameters.get_ibm4_iteration_count(ThotWordAlignmentModelType.FAST_ALIGN) == 0


def test_get_ibm4_iteration_count_set() -> None:
    parameters = ThotWordAlignmentParameters(ibm4_iteration_count=2)
    assert parameters.get_ibm4_iteration_count(ThotWordAlignmentModelType.IBM4) == 2
    assert parameters.get_ibm4_iteration_count(ThotWordAlignmentModelType.FAST_ALIGN) == 0
