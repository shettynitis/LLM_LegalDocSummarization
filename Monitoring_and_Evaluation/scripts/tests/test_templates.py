TEMPLATES = [
    "Explain clause 4.3 in plain English:\n{TXT}",
    "Write a 30‑word abstract of the following judgement:\n{TXT}",
    "Summarize the key holding:\n{TXT}",
]
REQUIRED_PASS = 0.80

def test_templates(predict, test_data):
    inputs, _ = test_data
    good=0; total=0
    for tpl in TEMPLATES:
        for txt in inputs[:20]:
            total+=1
            out = predict(tpl.replace("{TXT}", txt[:400]))
            if out and len(out.split())>5: good+=1
    assert good/total >= REQUIRED_PASS, f"{good}/{total} template passes"
