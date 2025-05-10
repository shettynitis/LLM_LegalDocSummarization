const inputEl      = document.getElementById("inputText");
const btnSummarize = document.getElementById("summarizeBtn");
const resultCard   = document.getElementById("result");
const summaryPre   = document.getElementById("summaryText");
const btnDownload  = document.getElementById("downloadTxtBtn");

btnSummarize.addEventListener("click", async () => {
  const text = inputEl.value.trim();
  if (!text) {
    return alert("Please paste some text first.");
  }

  // fire off the POST
  const res = await fetch("/summarize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    return alert(err.error || "Something went wrong.");
  }

  // show the summary and un-hide the card
  const { summary } = await res.json();
  summaryPre.textContent = summary;
  resultCard.classList.remove("d-none");
});

btnDownload.addEventListener("click", () => {
  const summary = summaryPre.textContent;
  const blob    = new Blob([summary], { type: "text/plain" });
  const url     = URL.createObjectURL(blob);
  const a       = document.createElement("a");
  a.href        = url;
  a.download    = "summary.txt";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});