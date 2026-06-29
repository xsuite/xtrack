from xtrack.general import parse_anchor_spec


def test_parse_anchor_spec():
    assert parse_anchor_spec("mq.1") == ("mq.1", None)
    assert parse_anchor_spec("mq.1", default_anchor="start") == ("mq.1", "start")
    assert parse_anchor_spec("mq.1@end") == ("mq.1", "end")
    assert parse_anchor_spec("mq.1@centre") == ("mq.1", "centre")

    try:
        parse_anchor_spec("mq.1@exit")
    except ValueError as error:
        assert "Invalid anchor" in str(error)
    else:
        raise AssertionError("Expected parse_anchor_spec to reject invalid anchors.")
