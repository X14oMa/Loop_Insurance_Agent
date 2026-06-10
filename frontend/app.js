const API = "";
const STORAGE_KEY = "insurance_agent_sessions";
const STORAGE_BOOT_KEY = "insurance_agent_server_boot_id";
/** 流式 DOM 刷新间隔（毫秒），避免每个 SSE 包全量 innerHTML */
const STREAM_THROTTLE_MS = 64;
/** 输入框自动增高上限（与 styles.css max-height 一致） */
const INPUT_MAX_HEIGHT = 160;
const WELCOME_HTML = `
  <div class="message message--assistant">
    <div class="message__avatar">AI</div>
    <div class="message__bubble">
      <p>您好！我是您的保险智能顾问。</p>
      <p>请先上传 PDF 保单，然后您可以询问保障范围、等待期、理赔条件、免责条款等问题。</p>
    </div>
  </div>
`;

const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const uploadProgress = document.getElementById("uploadProgress");
const progressBar = document.getElementById("progressBar");
const uploadStatus = document.getElementById("uploadStatus");
const policyList = document.getElementById("policyList");
const refreshBtn = document.getElementById("refreshBtn");
const clearPoliciesBtn = document.getElementById("clearPoliciesBtn");
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const inputCharHint = document.getElementById("inputCharHint");
const sendBtn = document.getElementById("sendBtn");
const statusBadge = document.getElementById("statusBadge");
const toast = document.getElementById("toast");
const sessionTabs = document.getElementById("sessionTabs");
const newSessionBtn = document.getElementById("newSessionBtn");
const clearChatBtn = document.getElementById("clearChatBtn");
const restartBanner = document.getElementById("restartBanner");
const restartBannerCloseBtn = document.getElementById("restartBannerCloseBtn");
const restartBannerClearBtn = document.getElementById("restartBannerClearBtn");

let isUploading = false;
let isChatting = false;
let pageUnloading = false;
/** 当前进行中的 chat/stream 请求，页面卸载时 abort */
let activeChatAbort = null;
let restartBannerVisible = false;
let sessions = [];
let activeSessionId = null;
let uiConfig = {
  disclaimer: "",
  source_notice: "",
};

function applyGlobalDisclaimer() {
  const el = document.getElementById("chatDisclaimer");
  if (!el) return;
  el.textContent = uiConfig.disclaimer || "";
}

async function loadUiConfig() {
  try {
    const res = await fetch(`${API}/api/ui-config`);
    if (!res.ok) return;
    const data = await res.json();
    uiConfig = {
      disclaimer: data.disclaimer || "",
      source_notice: data.source_notice || "",
    };
    applyGlobalDisclaimer();
  } catch {
    /* 使用空文案，避免阻塞页面 */
  }
}

function buildComplianceFootnote(compliance) {
  if (!compliance?.show_source_notice || !uiConfig.source_notice) {
    return "";
  }
  const hint = compliance.has_source_citation
    ? "以上涉及条款处已在正文中标注来源。"
    : "";
  const text = hint
    ? `${hint} ${uiConfig.source_notice}`
    : uiConfig.source_notice;
  return `<footer class="message__compliance" role="note">${escapeHtml(text)}</footer>`;
}

function showToast(message, type = "info") {
  toast.textContent = message;
  toast.className = `toast ${type}`;
  toast.classList.remove("hidden");
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => toast.classList.add("hidden"), 3200);
}

function formatDate(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

/** 无 Markdown 库时的兜底（仅转义 + 粗体 + 分段） */
function renderPlainText(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .split("\n")
    .map((line) => (line.trim() ? `<p>${line}</p>` : ""))
    .join("");
}

let markdownConfigured = false;

function configureMarkdown() {
  if (markdownConfigured || typeof marked === "undefined") {
    return;
  }
  marked.use({
    breaks: true,
    gfm: true,
    renderer: {
      link({ href, title, text }) {
        const safeHref = escapeHtml(href || "");
        const safeText = escapeHtml(text || href || "");
        const titleAttr = title ? ` title="${escapeHtml(title)}"` : "";
        return `<a href="${safeHref}"${titleAttr} target="_blank" rel="noopener noreferrer">${safeText}</a>`;
      },
    },
  });
  markdownConfigured = true;
}

/** 流式结束后再调用：完整 Markdown → 消毒后的 HTML */
function renderMarkdown(text) {
  const raw = String(text ?? "");
  if (!raw.trim()) {
    return "";
  }
  if (typeof marked === "undefined" || typeof DOMPurify === "undefined") {
    return renderPlainText(raw);
  }
  configureMarkdown();
  const html = marked.parse(raw);
  return DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true },
    ADD_ATTR: ["target", "rel"],
  });
}

function applyMarkdownBody(el, text) {
  el.classList.add("markdown-body");
  el.innerHTML = text.trim()
    ? renderMarkdown(text)
    : "";
}

/**
 * 增量流式文本：仅用 textContent + delta 追加，避免反复 innerHTML 卡死主线程。
 */
function createStreamTextController(container, cssClass) {
  const state = {
    streamId: null,
    pre: null,
    text: "",
    flushTimer: null,
    scrollEl: null,
    blocks: [],
  };

  function ensureStreamPre(streamId) {
    if (state.streamId === streamId && state.pre) {
      return;
    }
    state.streamId = streamId;
    state.text = "";
    state.pre = document.createElement("pre");
    state.pre.className = cssClass;
    container.appendChild(state.pre);
  }

  function flush() {
    if (state.pre) {
      state.pre.textContent = state.text;
    }
    if (state.scrollEl) {
      state.scrollEl.scrollTop = state.scrollEl.scrollHeight;
    }
  }

  function scheduleFlush() {
    if (state.flushTimer !== null) {
      return;
    }
    state.flushTimer = window.setTimeout(() => {
      state.flushTimer = null;
      flush();
    }, STREAM_THROTTLE_MS);
  }

  return {
    setScrollTarget(el) {
      state.scrollEl = el;
    },
    /** 工具调用等非流式块，整段追加一次 */
    appendBlock(text) {
      if (!text) return;
      state.blocks.push(text);
      const div = document.createElement("div");
      div.className = cssClass === "thinking-stream" ? "thinking-step" : "thinking-step";
      div.textContent = text;
      container.appendChild(div);
      scheduleFlush();
    },
    onStreamEvent(event) {
      const streamId = event.stream_id || "default";
      ensureStreamPre(streamId);
      let changed = false;
      if (event.delta) {
        state.text += event.delta;
        changed = true;
      } else if (event.content !== undefined && event.content !== null) {
        state.text = event.content;
        changed = true;
      }
      if (changed) {
        scheduleFlush();
      }
    },
    getCombinedText() {
      const streamPart = state.text.trim();
      const blockPart = state.blocks.join("\n\n").trim();
      if (streamPart && blockPart) {
        return `${blockPart}\n\n${streamPart}`;
      }
      return streamPart || blockPart;
    },
    flushNow() {
      if (state.flushTimer !== null) {
        clearTimeout(state.flushTimer);
        state.flushTimer = null;
      }
      flush();
    },
    reset() {
      if (state.flushTimer !== null) {
        clearTimeout(state.flushTimer);
        state.flushTimer = null;
      }
      state.streamId = null;
      state.pre = null;
      state.text = "";
      state.blocks = [];
      container.replaceChildren();
    },
  };
}

function createSession(title) {
  return {
    id: crypto.randomUUID(),
    title: title || `对话 ${sessions.length + 1}`,
    messagesHtml: WELCOME_HTML,
  };
}

function saveSessions() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  if (activeSessionId) {
    localStorage.setItem(`${STORAGE_KEY}_active`, activeSessionId);
  }
}

function loadSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    sessions = raw ? JSON.parse(raw) : [];
  } catch {
    sessions = [];
  }
  activeSessionId = localStorage.getItem(`${STORAGE_KEY}_active`);
  if (!sessions.length) {
    const session = createSession("对话 1");
    sessions.push(session);
    activeSessionId = session.id;
  }
  if (!sessions.find((s) => s.id === activeSessionId)) {
    activeSessionId = sessions[0].id;
  }
}

function getActiveSession() {
  return sessions.find((s) => s.id === activeSessionId);
}

function captureMessagesHtml() {
  if (restartBanner?.parentElement === chatMessages) {
    restartBanner.remove();
  }
  return chatMessages.innerHTML;
}

/** 保存前合并其它标签页已写入 localStorage 的会话，避免旧内存覆盖新记录 */
function mergeSessionsBeforeSave(activeId, capturedHtml) {
  let stored = [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      stored = JSON.parse(raw);
    }
  } catch {
    stored = [];
  }

  const memoryById = new Map(sessions.map((s) => [s.id, s]));
  const merged = new Map();

  for (const item of stored) {
    merged.set(item.id, item);
  }
  for (const item of sessions) {
    if (!merged.has(item.id)) {
      merged.set(item.id, item);
    }
  }

  const active = memoryById.get(activeId) || merged.get(activeId);
  if (active) {
    merged.set(activeId, { ...active, messagesHtml: capturedHtml });
  }

  sessions = Array.from(merged.values());
}

function persistActiveMessages() {
  const activeId = activeSessionId;
  const session = getActiveSession();
  if (!session) {
    return;
  }

  const capturedHtml = captureMessagesHtml();
  mergeSessionsBeforeSave(activeId, capturedHtml);

  try {
    saveSessions();
  } catch (err) {
    console.warn("会话保存失败:", err);
  }

  if (restartBannerVisible) {
    placeRestartBannerAtEnd();
  }
}

function persistActiveMessagesDeferred() {
  persistActiveMessages();
}

function renderSessionTabs() {
  sessionTabs.innerHTML = sessions
    .map(
      (s) => `
    <button type="button"
      class="session-tab ${s.id === activeSessionId ? "session-tab--active" : ""}"
      data-id="${s.id}">
      <span>${escapeHtml(s.title)}</span>
      ${sessions.length > 1 ? `<span class="session-tab__close" data-close="${s.id}">×</span>` : ""}
    </button>
  `,
    )
    .join("");

  sessionTabs.querySelectorAll(".session-tab").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      if (e.target.classList.contains("session-tab__close")) return;
      switchSession(btn.dataset.id);
    });
  });

  sessionTabs.querySelectorAll(".session-tab__close").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      closeSession(btn.dataset.close);
    });
  });
}

function renderChatFromSession() {
  const session = getActiveSession();
  chatMessages.innerHTML = session?.messagesHtml || WELCOME_HTML;
  if (restartBannerVisible) {
    placeRestartBannerAtEnd();
  } else {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
}

function switchSession(sessionId) {
  if (sessionId === activeSessionId) return;
  persistActiveMessages();
  activeSessionId = sessionId;
  renderSessionTabs();
  renderChatFromSession();
  saveSessions();
}

function closeSession(sessionId) {
  if (sessions.length <= 1) {
    showToast("至少保留一个对话窗口", "error");
    return;
  }
  const idx = sessions.findIndex((s) => s.id === sessionId);
  if (idx === -1) return;
  sessions.splice(idx, 1);
  if (activeSessionId === sessionId) {
    activeSessionId = sessions[Math.max(0, idx - 1)].id;
  }
  fetch(`${API}/api/sessions/${sessionId}`, { method: "DELETE" }).catch(() => {});
  renderSessionTabs();
  renderChatFromSession();
  saveSessions();
}

function newSession() {
  persistActiveMessages();
  const session = createSession(`对话 ${sessions.length + 1}`);
  sessions.push(session);
  activeSessionId = session.id;
  renderSessionTabs();
  renderChatFromSession();
  saveSessions();
}

function appendUserMessage(text) {
  const div = document.createElement("div");
  div.className = "message message--user";
  div.innerHTML = `
    <div class="message__avatar">我</div>
    <div class="message__bubble">${renderPlainText(text)}</div>
  `;
  chatMessages.appendChild(div);
  if (restartBannerVisible) {
    placeRestartBannerAtEnd();
  } else {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
  persistActiveMessages();
}

function createStreamingAssistantMessage() {
  const wrapper = document.createElement("div");
  wrapper.className = "message message--assistant message--streaming";
  wrapper.innerHTML = `
    <div class="message__avatar">AI</div>
    <div class="message__content">
      <details class="thinking-panel" open>
        <summary>思考过程 <span class="thinking-panel__status">进行中...</span></summary>
        <div class="thinking-panel__body"></div>
      </details>
      <div class="answer-panel">
        <div class="answer-panel__body"></div>
        <div class="message__compliance-slot"></div>
      </div>
    </div>
  `;
  chatMessages.appendChild(wrapper);
  if (restartBannerVisible) {
    placeRestartBannerAtEnd();
  } else {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
  const thinkingBody = wrapper.querySelector(".thinking-panel__body");
  const answerBody = wrapper.querySelector(".answer-panel__body");
  const thinkingController = createStreamTextController(thinkingBody, "thinking-stream");
  const answerController = createStreamTextController(answerBody, "answer-stream");
  thinkingController.setScrollTarget(chatMessages);
  answerController.setScrollTarget(chatMessages);

  return {
    wrapper,
    thinkingPanel: wrapper.querySelector(".thinking-panel"),
    thinkingBody,
    thinkingStatus: wrapper.querySelector(".thinking-panel__status"),
    answerBody,
    complianceSlot: wrapper.querySelector(".message__compliance-slot"),
    thinkingController,
    answerController,
  };
}

function finalizeStreamingMessage(ui, thinkingText, answerText, compliance) {
  if (ui.thinkingController) {
    ui.thinkingController.flushNow();
  }
  if (ui.answerController) {
    ui.answerController.flushNow();
  }
  if (thinkingText.trim()) {
    applyMarkdownBody(ui.thinkingBody, thinkingText);
  } else {
    ui.thinkingBody.classList.remove("markdown-body");
    ui.thinkingBody.innerHTML = "<p class=\"muted\">本次未产生额外推理步骤</p>";
  }
  applyMarkdownBody(ui.answerBody, answerText);
  if (ui.complianceSlot) {
    ui.complianceSlot.innerHTML = buildComplianceFootnote(compliance);
  }
  ui.thinkingStatus.textContent = "已完成";
  ui.thinkingPanel.open = false;
  ui.wrapper.classList.remove("message--streaming");
  persistActiveMessagesDeferred();
}

function hasStaleChatHistory() {
  return sessions.some((s) => s.messagesHtml?.includes("message--user"));
}

function dockRestartBanner() {
  if (!restartBanner) return;
  restartBanner.classList.add("hidden");
  if (restartBanner.parentElement !== document.body) {
    document.body.appendChild(restartBanner);
  }
}

function placeRestartBannerAtEnd() {
  if (!restartBanner || !restartBannerVisible) return;
  restartBanner.classList.remove("hidden");
  chatMessages.appendChild(restartBanner);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showRestartBanner() {
  restartBannerVisible = true;
  placeRestartBannerAtEnd();
}

function hideRestartBanner() {
  restartBannerVisible = false;
  dockRestartBanner();
}

function handleServerBootChange(serverBootId) {
  if (!serverBootId) return;
  const previous = localStorage.getItem(STORAGE_BOOT_KEY);
  if (previous && previous !== serverBootId && hasStaleChatHistory()) {
    showRestartBanner();
  }
  localStorage.setItem(STORAGE_BOOT_KEY, serverBootId);
}

async function checkHealth() {
  try {
    const res = await fetch(`${API}/api/health`);
    if (!res.ok) throw new Error("服务不可用");
    const data = await res.json();
    handleServerBootChange(data.server_boot_id);
    statusBadge.className = "header__status online";
    statusBadge.querySelector("span:last-child").textContent =
      `已连接 · ${data.policies_count} 份保单`;
    sendBtn.disabled = false;
    return true;
  } catch {
    statusBadge.className = "header__status error";
    statusBadge.querySelector("span:last-child").textContent = "服务未连接";
    sendBtn.disabled = true;
    return false;
  }
}

async function loadPolicies() {
  try {
    const res = await fetch(`${API}/api/documents`);
    if (!res.ok) throw new Error("加载失败");
    const data = await res.json();
    renderPolicies(data.documents || []);
  } catch (err) {
    showToast(err.message, "error");
  }
}

function renderPolicies(documents) {
  if (!documents.length) {
    policyList.innerHTML = '<li class="policy-list__empty">暂无保单，请先上传 PDF</li>';
    return;
  }

  policyList.innerHTML = documents
    .map(
      (doc) => `
    <li class="policy-item">
      <div class="policy-item__icon">PDF</div>
      <div class="policy-item__info">
        <div class="policy-item__name">${escapeHtml(doc.filename)}</div>
        <div class="policy-item__meta">${doc.size_kb} KB · ${formatDate(doc.uploaded_at)}</div>
      </div>
      <button type="button" class="policy-item__delete" data-filename="${escapeHtml(doc.filename)}" title="删除">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M18 6L6 18M6 6l12 12" stroke-linecap="round"/>
        </svg>
      </button>
    </li>
  `,
    )
    .join("");

  policyList.querySelectorAll(".policy-item__delete").forEach((btn) => {
    btn.addEventListener("click", () => deletePolicy(btn.dataset.filename));
  });
}

async function deletePolicy(filename) {
  if (!confirm(`确定删除「${filename}」吗？`)) return;
  try {
    const res = await fetch(`${API}/api/documents/${encodeURIComponent(filename)}`, {
      method: "DELETE",
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "删除失败");
    showToast("保单已删除", "success");
    await loadPolicies();
    await checkHealth();
  } catch (err) {
    showToast(err.message, "error");
  }
}

async function clearPolicyLibrary() {
  if (!confirm("确定清空全部保单记忆库吗？\n将删除所有 PDF 文件、向量索引和长期对话记忆。")) {
    return;
  }
  try {
    const res = await fetch(`${API}/api/documents`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clear_long_term: true }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "清空失败");
    showToast(
      `已清空：${data.deleted_files} 个文件，${data.deleted_memories} 条记忆`,
      "success",
    );
    await loadPolicies();
    await checkHealth();
  } catch (err) {
    showToast(err.message, "error");
  }
}

async function clearCurrentChat() {
  if (!confirm("确定清空当前对话窗口的所有消息吗？")) return;
  try {
    await fetch(`${API}/api/sessions/${activeSessionId}`, { method: "DELETE" });
    const session = getActiveSession();
    if (session) {
      session.messagesHtml = WELCOME_HTML;
      renderChatFromSession();
      saveSessions();
    }
    hideRestartBanner();
    showToast("当前对话已清空", "success");
  } catch (err) {
    showToast(err.message, "error");
  }
}

async function uploadFile(file) {
  if (isUploading) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showToast("仅支持 PDF 格式", "error");
    return;
  }

  isUploading = true;
  uploadProgress.classList.remove("hidden");
  progressBar.style.width = "20%";
  uploadStatus.textContent = `正在上传 ${file.name}...`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    progressBar.style.width = "50%";
    uploadStatus.textContent = "正在解析并建立索引...";
    const res = await fetch(`${API}/api/upload`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "上传失败");

    progressBar.style.width = "100%";
    uploadStatus.textContent = "索引完成";
    showToast(`「${data.filename}」已上传，共 ${data.chunks} 个片段`, "success");
    await loadPolicies();
    await checkHealth();

    const session = getActiveSession();
    if (session && session.messagesHtml === WELCOME_HTML) {
      session.title = file.name.replace(/\.pdf$/i, "").slice(0, 12) || session.title;
      renderSessionTabs();
    }

    appendUserMessage(`[系统] 已上传保单：${data.filename}`);
    const ui = createStreamingAssistantMessage();
    finalizeStreamingMessage(
      ui,
      "",
      `已成功导入保单「${data.filename}」，共解析 ${data.chunks} 个条款片段。您现在可以针对该保单提问了。`,
      null,
    );
  } catch (err) {
    showToast(err.message, "error");
    uploadStatus.textContent = "上传失败";
  } finally {
    isUploading = false;
    fileInput.value = "";
    setTimeout(() => {
      uploadProgress.classList.add("hidden");
      progressBar.style.width = "0";
    }, 1200);
  }
}

async function sendMessage(text) {
  if (isChatting || !text.trim()) return;
  await checkHealth();
  if (sendBtn.disabled) return;

  isChatting = true;
  sendBtn.disabled = true;
  appendUserMessage(text);
  messageInput.value = "";
  resetMessageInput();

  const ui = createStreamingAssistantMessage();
  let thinkingText = "";
  let answerText = "";
  const abortController = new AbortController();
  activeChatAbort = abortController;

  try {
    const res = await fetch(`${API}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: activeSessionId }),
      signal: abortController.signal,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "请求失败");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data: ")) continue;
        const event = JSON.parse(line.slice(6));

        if (event.type === "thinking") {
          if (event.stream) {
            ui.thinkingController.onStreamEvent(event);
          } else {
            ui.thinkingController.appendBlock(event.content || "");
          }
        } else if (event.type === "answer") {
          if (event.stream) {
            ui.answerController.onStreamEvent(event);
            answerText = event.content || answerText;
          } else if (event.content) {
            answerText = event.content;
            ui.answerController.onStreamEvent({
              stream_id: event.stream_id || "answer",
              content: event.content,
              delta: event.content,
              stream: true,
            });
          }
        } else if (event.type === "done") {
          answerText = event.answer || answerText;
          thinkingText = event.thinking || ui.thinkingController.getCombinedText();
          finalizeStreamingMessage(ui, thinkingText, answerText, event.compliance);
        } else if (event.type === "error") {
          throw new Error(event.message);
        }
      }
    }

    if (ui.wrapper.classList.contains("message--streaming")) {
      thinkingText = ui.thinkingController.getCombinedText();
      answerText = answerText || ui.answerController.getCombinedText() || "（无回答）";
      finalizeStreamingMessage(ui, thinkingText, answerText, null);
    }
  } catch (err) {
    if (!ui.wrapper.classList.contains("message--streaming")) {
      return;
    }
    const partialAnswer = answerText || ui.answerController.getCombinedText();
    const partialThinking = ui.thinkingController.getCombinedText();
    const benign = isBenignStreamError(err) || pageUnloading;

    if (partialAnswer.trim()) {
      finalizeStreamingMessage(ui, partialThinking, partialAnswer, null);
    } else if (benign) {
      ui.wrapper.remove();
      persistActiveMessages();
    } else {
      finalizeStreamingMessage(
        ui,
        partialThinking,
        `抱歉，出现了错误：${err.message}`,
        null,
      );
    }
  } finally {
    if (activeChatAbort === abortController) {
      activeChatAbort = null;
    }
    isChatting = false;
    sendBtn.disabled = false;
    messageInput.focus();
  }
}

uploadZone.addEventListener("click", () => fileInput.click());
browseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
});
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
});

refreshBtn.addEventListener("click", loadPolicies);
clearPoliciesBtn.addEventListener("click", clearPolicyLibrary);
newSessionBtn.addEventListener("click", newSession);
clearChatBtn.addEventListener("click", clearCurrentChat);
if (restartBannerCloseBtn) {
  restartBannerCloseBtn.addEventListener("click", hideRestartBanner);
}
if (restartBannerClearBtn) {
  restartBannerClearBtn.addEventListener("click", () => clearCurrentChat());
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(messageInput.value);
});

messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage(messageInput.value);
  }
});

function updateInputCharHint() {
  if (!inputCharHint) {
    return;
  }
  const len = messageInput.value.length;
  const max = Number(messageInput.maxLength) || 4000;
  if (len <= 0) {
    inputCharHint.textContent = "";
    return;
  }
  inputCharHint.textContent = `${len} / ${max} 字 · Shift+Enter 换行`;
}

function resizeMessageInput() {
  messageInput.style.height = "auto";
  const nextHeight = Math.min(messageInput.scrollHeight, INPUT_MAX_HEIGHT);
  messageInput.style.height = `${nextHeight}px`;
  messageInput.style.overflowY =
    messageInput.scrollHeight > INPUT_MAX_HEIGHT ? "auto" : "hidden";
  updateInputCharHint();
}

function resetMessageInput() {
  messageInput.style.height = "auto";
  messageInput.style.overflowY = "hidden";
  updateInputCharHint();
}

messageInput.addEventListener("input", resizeMessageInput);
messageInput.addEventListener("paste", () => {
  requestAnimationFrame(resizeMessageInput);
});
resetMessageInput();

function isBenignStreamError(err) {
  if (!err) {
    return false;
  }
  if (err.name === "AbortError") {
    return true;
  }
  const msg = String(err.message || err).toLowerCase();
  return (
    msg.includes("network error") ||
    msg.includes("failed to fetch") ||
    msg.includes("load failed") ||
    msg.includes("the user aborted")
  );
}

function flushMessagesBeforeUnload() {
  pageUnloading = true;
  activeChatAbort?.abort();
  if (!getActiveSession()) {
    return;
  }
  persistActiveMessages();
}

window.addEventListener("pagehide", flushMessagesBeforeUnload);
window.addEventListener("beforeunload", flushMessagesBeforeUnload);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") {
    flushMessagesBeforeUnload();
  }
});

window.addEventListener("storage", (e) => {
  if (e.key !== STORAGE_KEY && e.key !== `${STORAGE_KEY}_active`) {
    return;
  }
  if (isChatting) {
    return;
  }
  loadSessions();
  renderSessionTabs();
  renderChatFromSession();
});

(async function init() {
  loadSessions();
  renderSessionTabs();
  renderChatFromSession();
  await loadUiConfig();
  await checkHealth();
  await loadPolicies();
})();
