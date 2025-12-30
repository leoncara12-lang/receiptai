async function api(path, opts={}) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function setMsg(el, msg) {
  el.textContent = msg || "";
}

function statusLabel(s) {
  if (s === "GREEN") return "✅ VERDE (supportato)";
  if (s === "YELLOW") return "🟡 GIALLO (parziale)";
  return "🔴 ROSSO (non supportato)";
}

function claimCard(c) {
  const div = document.createElement("div");
  div.className = "claim";
  div.innerHTML = `
    <div class="row">
      <span class="chip"><b>${c.id}</b></span>
      <span class="chip">type: ${c.type}</span>
      <span class="chip">risk: ${c.risk}</span>
      <span class="chip">verdict: <b>${c.verdict}</b></span>
      <span class="chip">conf: ${Number(c.confidence).toFixed(2)}</span>
    </div>
    <div style="margin-top:10px; line-height:1.4">${c.text}</div>
    <hr />
    <div class="small"><b>Evidence</b></div>
  `;
  if (c.evidence && c.evidence.length) {
    const evWrap = document.createElement("div");
    evWrap.style.display = "grid";
    evWrap.style.gap = "8px";
    evWrap.style.marginTop = "8px";
    c.evidence.slice(0,3).forEach(e => {
      const ev = document.createElement("div");
      ev.className = "claim";
      ev.style.background = "#0b0f19";
      ev.innerHTML = `
        <div class="small">${e.source_id} · ${e.locator} · sim ${Number(e.similarity).toFixed(2)}</div>
        <pre class="small">${e.excerpt}</pre>
      `;
      evWrap.appendChild(ev);
    });
    div.appendChild(evWrap);
  } else {
    const p = document.createElement("div");
    p.className = "small";
    p.style.marginTop = "6px";
    p.textContent = "Nessuna evidenza trovata.";
    div.appendChild(p);
  }
  return div;
}

async function refreshDocs() {
  const docsEl = document.getElementById("docs");
  docsEl.innerHTML = "";
  try {
    const docs = await api("/documents");
    if (!docs.length) {
      docsEl.innerHTML = '<span class="small">Nessun documento ancora.</span>';
      return;
    }
    docs.forEach(d => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = `#${d.id} ${d.title}`;
      docsEl.appendChild(chip);
    });
  } catch (e) {
    docsEl.innerHTML = '<span class="small">Errore nel caricare i documenti.</span>';
  }
}

document.getElementById("uploadBtn").addEventListener("click", async () => {
  const title = document.getElementById("docTitle").value.trim();
  const file = document.getElementById("docFile").files[0];
  const msg = document.getElementById("uploadMsg");
  setMsg(msg, "");
  if (!title || !file) { setMsg(msg, "Scrivi un titolo e scegli un file."); return; }

  const fd = new FormData();
  fd.append("file", file);

  try {
    const res = await api(`/documents/upload?title=${encodeURIComponent(title)}`, { method: "POST", body: fd });
    setMsg(msg, `Caricato: ${res.title} (id ${res.id})`);
    await refreshDocs();
  } catch (e) {
    setMsg(msg, "Errore upload: " + e.message);
  }
});

document.getElementById("verifyBtn").addEventListener("click", async () => {
  const text = document.getElementById("text").value;
  const policy = document.getElementById("policy").value;
  const topK = Number(document.getElementById("topK").value || 5);
  const msg = document.getElementById("verifyMsg");
  setMsg(msg, "");
  const result = document.getElementById("result");
  result.classList.add("hidden");

  try {
    const res = await api("/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, policy_profile: policy, top_k: topK })
    });

    document.getElementById("status").textContent = statusLabel(res.overall_status);
    document.getElementById("receiptId").textContent = `receipt_id: ${res.receipt_id}`;
    document.getElementById("output").textContent = res.output_text;

    const claimsEl = document.getElementById("claims");
    claimsEl.innerHTML = "";
    res.claims.forEach(c => claimsEl.appendChild(claimCard(c)));

    result.classList.remove("hidden");
  } catch (e) {
    setMsg(msg, "Errore verifica: " + e.message);
  }
});

refreshDocs();
