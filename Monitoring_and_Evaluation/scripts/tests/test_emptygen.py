def test_no_empty(predictions):
    assert all(p.strip() for p in predictions), "Empty summary generated"