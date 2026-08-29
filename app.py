import gradio as gr

from dotenv import load_dotenv
from answer import answer_question


load_dotenv(override=True)


# ============================================================
# Configuration
# ============================================================

APP_TITLE = "Insurellm Expert Assistant"

MODEL = "qwen3:4b"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
RETRIEVAL_K = 5


# ============================================================
# Custom CSS — Premium Dark UI
# ============================================================

CSS = """
/* =========================================================
   Design Tokens
   ========================================================= */

:root {
    --primary: #f97316;
    --primary-dark: #ea580c;
    --primary-glow: rgba(249, 115, 22, 0.15);
    --bg: #0b1120;
    --surface: #151e32;
    --surface-hover: #1e293b;
    --border: #273656;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --success: #22c55e;
    --radius: 14px;
    --radius-sm: 10px;
}

/* =========================================================
   Global Overrides
   ========================================================= */

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    color: var(--text) !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Hide default Gradio footer */
footer { display: none !important; }

/* =========================================================
   Header
   ========================================================= */

.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    background: linear-gradient(90deg, rgba(11,17,32,0.95), rgba(21,30,50,0.95));
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(16px);
    position: sticky;
    top: 0;
    z-index: 50;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: 0 0 20px var(--primary-glow);
}

.brand-title {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text);
}

.brand-sub {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 1px;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    color: var(--success);
}

.pulse {
    width: 6px;
    height: 6px;
    background: var(--success);
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
    box-shadow: 0 0 6px var(--success);
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(0.85); }
}

/* =========================================================
   Main Grid Layout
   ========================================================= */

.main-grid {
    display: grid;
    grid-template-columns: 250px 1fr 320px;
    height: calc(100vh - 68px);
    overflow: hidden;
}

/* =========================================================
   Sidebar (Left)
   ========================================================= */

.sidebar-col {
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 18px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    overflow-y: auto;
}

.info-card {
    background: rgba(21, 30, 50, 0.6);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px;
    transition: all 0.25s ease;
}

.info-card:hover {
    border-color: var(--primary);
    box-shadow: 0 0 0 2px var(--primary-glow);
}

.info-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    margin-bottom: 8px;
}

.model-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    background: rgba(249, 115, 22, 0.08);
    border: 1px solid rgba(249, 115, 22, 0.18);
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    color: var(--primary);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

.stat-line {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid rgba(39, 54, 86, 0.5);
    font-size: 12px;
}

.stat-line:last-child { border-bottom: none; }

.stat-key { color: var(--text-muted); }
.stat-val { color: var(--text); font-weight: 600; }

.clear-btn-wrap {
    margin-top: auto;
    padding-top: 8px;
}

/* =========================================================
   Chat Area (Center)
   ========================================================= */

.chat-col {
    display: flex;
    flex-direction: column;
    background: var(--bg);
    position: relative;
    min-width: 0;
}

.chat-header-bar {
    padding: 12px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-muted);
}

/* Chatbot overrides */
#chatbot {
    flex: 1;
    border: none !important;
    background: transparent !important;
    border-radius: 0 !important;
}

#chatbot .message-wrap {
    padding: 10px 24px !important;
}

/* Make sure no floating label badge overlaps the placeholder/messages */
#chatbot .label-wrap,
#chatbot label,
#chatbot > .icon-button-wrapper {
    display: none !important;
}

#chatbot .placeholder {
    padding-top: 36px !important;
}

/* User bubble */
#chatbot .message.user {
    justify-content: flex-end !important;
}
#chatbot .message.user .bubble {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: #fff !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 10px 16px !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
    max-width: 75% !important;
    box-shadow: 0 4px 16px rgba(249, 115, 22, 0.25) !important;
}

/* Bot bubble */
#chatbot .message.bot .bubble {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 12px 18px !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
    max-width: 85% !important;
}

/* Avatars */
#chatbot .avatar {
    width: 30px !important;
    height: 30px !important;
    border-radius: 8px !important;
    font-size: 15px !important;
}

/* =========================================================
   Input Area
   ========================================================= */

.input-area {
    padding: 12px 24px 20px;
    background: linear-gradient(to top, var(--bg) 70%, transparent);
}

.suggestion-row {
    display: flex;
    gap: 8px;
    overflow-x: auto;
    padding-bottom: 10px;
    margin-bottom: 4px;
    scrollbar-width: none;
}
.suggestion-row::-webkit-scrollbar { display: none; }

.suggestion-pill {
    white-space: nowrap;
    padding: 6px 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    font-size: 12px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.2s;
}
.suggestion-pill:hover {
    background: rgba(249, 115, 22, 0.1);
    border-color: var(--primary);
    color: var(--primary);
}

.msg-input textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 12px 16px !important;
    color: var(--text) !important;
    font-size: 14px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25) !important;
    transition: all 0.2s !important;
}
.msg-input textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-glow), 0 4px 20px rgba(0,0,0,0.25) !important;
    outline: none !important;
}

.send-btn button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    height: 44px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(249, 115, 22, 0.3) !important;
    transition: all 0.2s !important;
}
.send-btn button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(249, 115, 22, 0.4) !important;
}

/* =========================================================
   Sources Panel (Right)
   ========================================================= */

.sources-col {
    background: var(--surface);
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 0;
}

.sources-header {
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.sources-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 6px;
}

.badge {
    background: var(--primary);
    color: #fff;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 10px;
}

/* Source cards injected via HTML */
.src-card {
    background: rgba(11, 17, 32, 0.5);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 12px;
    margin-bottom: 10px;
    transition: all 0.2s;
    cursor: default;
}
.src-card:hover {
    border-color: var(--primary);
    background: rgba(249, 115, 22, 0.04);
    transform: translateX(2px);
}
.src-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.src-rank {
    width: 22px;
    height: 22px;
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 700;
    flex-shrink: 0;
}
.src-name {
    font-size: 11px;
    font-weight: 600;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.src-body {
    font-size: 11px;
    color: var(--text-muted);
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.empty-state {
    text-align: center;
    padding: 50px 20px;
    color: var(--text-muted);
}
.empty-icon {
    font-size: 36px;
    margin-bottom: 14px;
    opacity: 0.4;
}
.empty-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 4px;
}
.empty-desc {
    font-size: 11px;
    opacity: 0.7;
}

/* =========================================================
   Scrollbars
   ========================================================= */

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
"""


# ============================================================
# Format Sources as rich HTML
# ============================================================

def format_sources(context):
    if not context:
        return """
        <div class="empty-state">
            <div class="empty-icon">📚</div>
            <div class="empty-title">No sources yet</div>
            <div class="empty-desc">Ask a question to retrieve relevant documents</div>
        </div>
        """

    html = f"""
    <div class="sources-header">
        <div class="sources-title">📚 Sources <span class="badge">{len(context)}</span></div>
    </div>
    <div style="padding: 14px; overflow-y: auto; flex: 1;">
    """

    for rank, doc in enumerate(context, start=1):
        source = doc.metadata.get("source", "Unknown source")
        source_name = source.replace("\\", "/").split("/")[-1]
        content = doc.page_content
        if len(content) > 280:
            content = content[:280] + "..."

        html += f"""
        <div class="src-card">
            <div class="src-header">
                <div class="src-rank">{rank}</div>
                <div class="src-name" title="{source_name}">{source_name}</div>
            </div>
            <div class="src-body">{content}</div>
        </div>
        """

    html += "</div>"
    return html


# ============================================================
# Chat Logic
# ============================================================

def chat(history):
    if not history:
        return history, format_sources([])

    last_message = history[-1]["content"]
    prior = history[:-1]

    try:
        answer, context = answer_question(last_message, prior)
    except Exception as e:
        print(f"[chat] Error: {e}")
        answer = (
            "⚠️ **Connection Error**\n\n"
            "I couldn't reach the knowledge base. Please check that:\n"
            "1. Ollama is running (`ollama serve`)\n"
            "2. The vector database exists (`preprocessed_db/`)\n"
            "3. Models are pulled (`qwen3:4b` & `qwen3-embedding:0.6b`)"
        )
        context = []

    history.append({"role": "assistant", "content": answer})
    return history, format_sources(context)


def put_message_in_chatbot(message, history):
    if not message.strip():
        return "", history
    history = history or []
    history.append({"role": "user", "content": message})
    return "", history


def clear_chat():
    return [], format_sources([])


def use_suggestion(question):
    return question


# ============================================================
# Main UI
# ============================================================

def main():
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(title=APP_TITLE, theme=theme, css=CSS) as demo:

        # -------------------- Header --------------------
        gr.HTML("""
        <div class="app-header">
            <div class="brand">
                <div class="brand-icon">🏢</div>
                <div>
                    <div class="brand-title">Insurellm Expert</div>
                    <div class="brand-sub">AI Knowledge Assistant</div>
                </div>
            </div>
            <div class="status-badge">
                <span class="pulse"></span>
                System Online
            </div>
        </div>
        """)

        # -------------------- Main Grid --------------------
        with gr.Row(equal_height=True, elem_classes="main-grid"):

            # ====== Sidebar ======
            with gr.Column(scale=1, min_width=250, elem_classes="sidebar-col"):
                gr.HTML("""
                <div class="info-card">
                    <div class="info-label">🤖 Generation</div>
                    <div class="model-tag">qwen3:4b</div>
                </div>
                <div class="info-card">
                    <div class="info-label">🔎 Embedding</div>
                    <div class="model-tag">qwen3-embedding:0.6b</div>
                </div>
                <div class="info-card">
                    <div class="info-label">⚙️ Settings</div>
                    <div class="stat-line"><span class="stat-key">Top K</span><span class="stat-val">5</span></div>
                    <div class="stat-line"><span class="stat-key">Temperature</span><span class="stat-val">0.0</span></div>
                    <div class="stat-line"><span class="stat-key">Chunk Size</span><span class="stat-val">800</span></div>
                </div>
                """)

                clear_btn = gr.Button(
                    "🗑️ Clear Chat",
                    variant="secondary",
                    elem_classes="clear-btn-wrap"
                )

            # ====== Chat ======
            with gr.Column(scale=4, elem_classes="chat-col"):
                gr.HTML('<div class="chat-header-bar">💬 Conversation</div>')

                chatbot = gr.Chatbot(
                    label=None,
                    show_label=False,
                    height=520,
                    type="messages",
                    show_copy_button=True,
                    elem_id="chatbot",
                    bubble_full_width=False,
                    placeholder=(
                        "👋 **Welcome to Insurellm!**\n\n"
                        "I'm your AI assistant for the company knowledge base. "
                        "Ask me about employees, projects, contracts, or performance."
                    ),
                )

                with gr.Row(elem_classes="input-area"):
                    with gr.Column(scale=6):
                        msg = gr.Textbox(
                            placeholder="Ask anything about Insurellm...",
                            show_label=False,
                            lines=1,
                            max_lines=4,
                            elem_classes="msg-input",
                        )
                    with gr.Column(scale=1, min_width=80):
                        send = gr.Button("➤ Send", variant="primary", elem_classes="send-btn")

                with gr.Row(elem_classes="suggestion-row"):
                    s1 = gr.Button("🏆 IIOTY 2023 winner?", size="sm", variant="secondary")
                    s2 = gr.Button("👔 Who is the CEO?", size="sm", variant="secondary")
                    s3 = gr.Button("👤 Maxine Thompson", size="sm", variant="secondary")
                    s4 = gr.Button("📊 Highest performance", size="sm", variant="secondary")

            # ====== Sources ======
            with gr.Column(scale=2, min_width=320, elem_classes="sources-col"):
                sources_html = gr.HTML(format_sources([]))

        # -------------------- Events --------------------
        msg.submit(
            put_message_in_chatbot,
            [msg, chatbot],
            [msg, chatbot]
        ).then(
            chat,
            [chatbot],
            [chatbot, sources_html]
        )

        send.click(
            put_message_in_chatbot,
            [msg, chatbot],
            [msg, chatbot]
        ).then(
            chat,
            [chatbot],
            [chatbot, sources_html]
        )

        for btn in [s1, s2, s3, s4]:
            btn.click(use_suggestion, [btn], [msg])

        clear_btn.click(clear_chat, outputs=[chatbot, sources_html])

    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()