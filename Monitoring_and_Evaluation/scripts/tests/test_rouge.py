from evaluate import load
ROUGE_MIN = 0.45

def test_rouge(predictions, test_data):
    _, refs = test_data
    rouge = load("rouge")
    score = rouge.compute(predictions=predictions, references=refs)["rougeL"]
    assert score >= ROUGE_MIN, f"ROUGE‑L {score:.3f} < {ROUGE_MIN}"