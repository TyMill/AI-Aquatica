from ai_aquatica import cli


def test_cli_version(capsys):
    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "AI-Aquatica" in out


def test_cli_describe_public_symbol(capsys):
    assert cli.main(["--describe", "WaterQualityPipeline"]) == 0
    out = capsys.readouterr().out
    assert "WaterQualityPipeline:" in out


def test_cli_rejects_unknown_symbol(capsys):
    assert cli.main(["--describe", "not_a_public_symbol"]) == 1
    err = capsys.readouterr().err
    assert "not part of the public API" in err


def test_cli_lists_exports(capsys):
    assert cli.main(["--list-exports"]) == 0
    out = capsys.readouterr().out
    assert "WaterQualityPipeline" in out
    assert "calculate_charge_balance" in out
