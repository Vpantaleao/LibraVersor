"""Configuração comum dos testes."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # permite `import libra_core`, `import main` etc.


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="roda também os testes lentos (treinam um modelo com os dados reais)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="teste lento: use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def root():
    return ROOT
