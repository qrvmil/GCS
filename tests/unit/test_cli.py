from online_gcs.cli import main


def test_zero_iterations_returns_usage_error(capsys) -> None:
    assert main(["--iterations", "0"]) == 2
    assert "iterations must be at least 1" in capsys.readouterr().err


def test_help_does_not_construct_drake_scene() -> None:
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
