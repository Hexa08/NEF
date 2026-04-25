from math import isclose

import nef


def assert_nested_close(actual, expected, tol=1e-6):
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_nested_close(a, e, tol)
    else:
        assert isclose(actual, expected, rel_tol=tol, abs_tol=tol)


def test_lazy_execution_and_materialization():
    a = nef.tensor([1, 2, 3], dtype=nef.float32)
    b = nef.tensor([4, 5, 6], dtype=nef.float32)
    c = nef.add(a, b)

    assert c._materialized is False

    out = c.execute().numpy()

    assert out == [5, 7, 9]
    assert c._materialized is True


def test_matmul_softmax_pipeline():
    a = nef.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = nef.tensor([[2.0, 0.0], [1.0, 2.0]])

    logits = nef.matmul(a, b)
    probs = nef.softmax(logits, dim=-1)

    expected_logits = [[4.0, 4.0], [10.0, 8.0]]
    expected_probs = [[0.5, 0.5], [0.8807970779778824, 0.11920292202211755]]

    assert logits.numpy() == expected_logits
    assert_nested_close(probs.numpy(), expected_probs)


def test_auto_device_assignment_cpu():
    x = nef.tensor([1.0, 2.0, 3.0])
    y = nef.tensor([2.0, 3.0, 4.0])
    z = nef.mul(x, y)

    assert z.device is None
    z.execute()
    assert z.device == "cpu"
