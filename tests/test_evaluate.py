from airwm.evaluate import evaluation_indices


def test_evaluation_covers_flights_and_late_windows():
    windows = [(episode, start) for episode in range(10) for start in range(0, 900, 60)]
    indices = evaluation_indices(windows, 100)
    assert len(indices) == len(set(indices)) == 100
    for episode in range(10):
        starts = [windows[i][1] for i in indices if windows[i][0] == episode]
        assert len(starts) == 10
        assert min(starts) == 0
        assert max(starts) == 840
    short = evaluation_indices(windows, 10)
    assert {windows[i][0] for i in short} == set(range(10))
    assert all(windows[i][1] == 420 for i in short)


def test_evaluation_handles_short_flights_and_oversized_budget():
    windows = [(0, 0), (1, 0), (1, 60), (1, 120), (2, 0), (2, 60)]
    assert sorted(evaluation_indices(windows, 100)) == list(range(len(windows)))
    selected = evaluation_indices(windows, 5)
    assert len(selected) == len(set(selected)) == 5
    assert {windows[i][0] for i in selected} == {0, 1, 2}
