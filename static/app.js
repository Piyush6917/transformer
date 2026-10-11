(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const toast = $("#toast");
  let toastTimer;
  function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 3000);
  }
  async function request(url, options) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "Something went wrong.");
    return data;
  }
  $$(".nav-item").forEach(button => button.addEventListener("click", () => {
    $$(".nav-item").forEach(item => item.classList.toggle("active", item === button));
    $$(".view").forEach(view => view.classList.toggle("active", view.id === button.dataset.view));
    $("#breadcrumb-title").textContent = button.dataset.title;
  }));

  const fileInput = $("#pdf-input"), dropzone = $("#dropzone");
  function openPicker() { fileInput.click(); }
  $("#upload-trigger").addEventListener("click", openPicker);
  $("#panel-upload").addEventListener("click", openPicker);
  dropzone.addEventListener("click", openPicker);
  fileInput.addEventListener("change", () => uploadFiles(fileInput.files));
  ["dragenter", "dragover"].forEach(name => dropzone.addEventListener(name, event => {
    event.preventDefault(); dropzone.classList.add("dragover");
  }));
  ["dragleave", "drop"].forEach(name => dropzone.addEventListener(name, event => {
    event.preventDefault(); dropzone.classList.remove("dragover");
  }));
  dropzone.addEventListener("drop", event => uploadFiles(event.dataTransfer.files));
  async function uploadFiles(files) {
    if (!files || !files.length) return;
    const form = new FormData();
    [...files].forEach(file => form.append("files", file));
    try {
      const data = await request("/api/upload", { method: "POST", body: form });
      showToast(data.saved.length + (data.saved.length === 1 ? " PDF added" : " PDFs added"));
      fileInput.value = "";
      await refreshStatus();
    } catch (error) { showToast(error.message); }
  }
  function renderDocuments(names) {
    const list = $("#document-list");
    list.replaceChildren();
    $("#pdf-count").textContent = String(names.length);
    $("#documents-total").textContent = String(names.length);
    if (!names.length) {
      const empty = document.createElement("div");
      empty.className = "empty-documents";
      empty.textContent = "Your PDFs will appear here";
      list.append(empty);
      return;
    }
    names.forEach(name => {
      const item = document.createElement("div");
      item.className = "document-item";
      const icon = document.createElement("span");
      icon.className = "pdf-icon";
      icon.textContent = "PDF";
      const meta = document.createElement("span");
      meta.className = "document-meta";
      const title = document.createElement("b");
      title.textContent = name;
      const note = document.createElement("small");
      note.textContent = "Ready to search";
      meta.append(title, note);
      item.append(icon, meta);
      list.append(item);
    });
  }
  async function refreshStatus() {
    try {
      const status = await request("/api/status");
      renderDocuments(status.pdf_names);
      $("#device-label").textContent = "Image model · " + status.device;
    } catch (_) { /* status is informational */ }
  }
  refreshStatus();

  const messages = $("#chat-messages"), questionInput = $("#question-input");
  function addMessage(text, isUser, sources = []) {
    const row = document.createElement("div");
    row.className = "message-row" + (isUser ? " user" : "");
    const avatar = document.createElement("span");
    avatar.className = "message-avatar";
    avatar.textContent = isUser ? "P" : "✳";
    const body = document.createElement("div");
    body.className = "message-body";
    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = text;
    body.append(bubble);
    if (sources.length) {
      const sourceRow = document.createElement("div");
      sourceRow.className = "message-sources";
      sources.forEach(source => {
        const tag = document.createElement("span");
        tag.className = "source-tag";
        tag.textContent = source.document + " · page " + source.page;
        sourceRow.append(tag);
      });
      body.append(sourceRow);
    }
    row.append(avatar, body);
    messages.append(row);
    messages.scrollTop = messages.scrollHeight;
    return row;
  }
  async function askQuestion(question) {
    const clean = question.trim();
    if (!clean) return;
    $(".welcome-message", messages)?.remove();
    addMessage(clean, true);
    questionInput.value = "";
    questionInput.style.height = "auto";
    const pending = addMessage("Looking through your documents…", false);
    $(".message-bubble", pending).classList.add("typing-bubble");
    try {
      const data = await request("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: clean })
      });
      pending.remove();
      addMessage(data.answer, false, data.sources || []);
    } catch (error) {
      pending.remove();
      addMessage(error.message, false);
    }
  }
  $("#chat-form").addEventListener("submit", event => {
    event.preventDefault();
    askQuestion(questionInput.value);
  });
  questionInput.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault(); $("#chat-form").requestSubmit();
    }
  });
  questionInput.addEventListener("input", () => {
    questionInput.style.height = "auto";
    questionInput.style.height = Math.min(questionInput.scrollHeight, 112) + "px";
  });
  $$(".question-chip").forEach(button => button.addEventListener("click", () => {
    questionInput.value = button.dataset.question; questionInput.focus();
  }));
  $("#clear-chat").addEventListener("click", () => {
    messages.replaceChildren();
    const welcome = document.createElement("div");
    welcome.className = "welcome-message";
    const icon = document.createElement("span"); icon.className = "welcome-icon"; icon.textContent = "✳";
    const title = document.createElement("h2"); title.textContent = "Ready for your next question";
    const text = document.createElement("p"); text.textContent = "Ask about the documents in your workspace and I’ll show the pages I used.";
    welcome.append(icon, title, text); messages.append(welcome);
  });

  const promptInput = $("#prompt-input");
  promptInput.addEventListener("input", () => $("#prompt-count").textContent = String(promptInput.value.length));
  $$(".prompt-ideas button").forEach(button => button.addEventListener("click", () => {
    promptInput.value = button.dataset.prompt;
    promptInput.dispatchEvent(new Event("input"));
    promptInput.focus();
  }));
  $("#steps-input").addEventListener("input", event => $("#steps-value").value = event.target.value);
  const generateButton = $("#generate-button");
  const emptyPreview = $("#preview-empty"), generatingState = $("#generating-state");
  const resultImage = $("#result-image"), downloadLink = $("#download-link");
  const generationTitle = $("#generation-title"), generationMessage = $("#generation-message");
  const generationSubtitle = $("#generation-subtitle"), caption = $("#preview-caption");
  function showGenerating(message, state) {
    emptyPreview.hidden = true; resultImage.hidden = true; downloadLink.hidden = true;
    generatingState.hidden = false;
    generationTitle.textContent = state === "loading" ? "Warming up the model" : "Creating your image";
    generationMessage.textContent = message || "Your prompt is being transformed into an image.";
    generationSubtitle.textContent = state === "loading" ? "First run downloads model weights" : "Image generation can take a few minutes on CPU";
    caption.textContent = "Your canvas is taking shape…";
  }
  async function pollJob(jobId) {
    while (true) {
      const job = await request("/api/jobs/" + encodeURIComponent(jobId));
      if (job.status === "complete") return job;
      if (job.status === "error") throw new Error(job.message || "Image generation failed.");
      showGenerating(job.message, job.status);
      await new Promise(resolve => setTimeout(resolve, 1600));
    }
  }
  generateButton.addEventListener("click", async () => {
    const prompt = promptInput.value.trim();
    if (!prompt) { showToast("Write a prompt first, like “a red car at sunset”."); promptInput.focus(); return; }
    generateButton.disabled = true;
    generateButton.querySelector("span:nth-child(2)").textContent = "Starting…";
    showGenerating("Getting your prompt ready.", "queued");
    try {
      const job = await request("/api/generate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          steps: Number($("#steps-input").value),
          guidance: 7.5,
          seed: Number($("#seed-input").value)
        })
      });
      const done = await pollJob(job.job_id);
      resultImage.src = done.image_url + "?t=" + Date.now();
      resultImage.hidden = false; generatingState.hidden = true; emptyPreview.hidden = true;
      downloadLink.href = done.download_url; downloadLink.hidden = false;
      caption.textContent = prompt.length > 62 ? prompt.slice(0, 59) + "…" : prompt;
      showToast("Image created");
    } catch (error) {
      generatingState.hidden = true; resultImage.hidden = true; emptyPreview.hidden = false;
      caption.textContent = "Generation did not finish";
      showToast(error.message);
    } finally {
      generateButton.disabled = false;
      generateButton.querySelector("span:nth-child(2)").textContent = "Create image";
    }
  });
  const modal = $("#about-modal");
  $("#about-button").addEventListener("click", () => modal.hidden = false);
  $("#close-modal").addEventListener("click", () => modal.hidden = true);
  modal.addEventListener("click", event => { if (event.target === modal) modal.hidden = true; });
  document.addEventListener("keydown", event => { if (event.key === "Escape") modal.hidden = true; });
})();