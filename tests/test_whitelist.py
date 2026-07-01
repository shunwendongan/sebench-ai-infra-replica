from sebench_infra.orchestrator.whitelist import PathWhitelist


def test_whitelist_accepts_allowed_submission_path() -> None:
    whitelist = PathWhitelist(["submission/"])

    assert whitelist.is_allowed("submission/answer.txt")


def test_whitelist_blocks_escape_and_absolute_path() -> None:
    whitelist = PathWhitelist(["submission/"])

    assert not whitelist.is_allowed("../submission/answer.txt")
    assert not whitelist.is_allowed("/tmp/answer.txt")
    assert not whitelist.is_allowed("other/answer.txt")
