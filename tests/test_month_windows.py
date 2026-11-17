from datetime import date
from src.ingest_statcast import windows

def test_seven_day_windows():
    assert list(windows(date(2025, 1, 30), date(2025, 2, 8), 7)) == [
        (date(2025, 1, 30), date(2025, 2, 5)),
        (date(2025, 2, 6), date(2025, 2, 8)),
    ]

def test_one_day_window():
    assert list(windows(date(2025, 4, 1), date(2025, 4, 1), 7)) == [
        (date(2025, 4, 1), date(2025, 4, 1))
    ]
