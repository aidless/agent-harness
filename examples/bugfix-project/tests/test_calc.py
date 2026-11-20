import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import calc


def test_add():
    assert calc.add(2, 3) == 5


def test_mul():
    assert calc.mul(3, 4) == 12


def test_div():
    assert calc.div(10, 2) == 5
