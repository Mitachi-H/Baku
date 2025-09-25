"""日本語UIの心理療法チャットボットモック."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import gradio as gr

from demo_mini_llm_cbt import FineTunedCBTSystem

SESSION_LOG_DIR = Path("artifacts/chat_sessions")
SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)

CBT_SYSTEM = FineTunedCBTSystem()


def build_reply(message: str) -> str:
    analysis = CBT_SYSTEM.process_dream(message)
    reply_lines = [analysis["response"].strip()]

    plan = analysis.get("cbt_plan", {})
    if plan.get("primary"):
        reply_lines.append("今日のワーク提案: " + "、".join(plan["primary"]))
    if plan.get("support"):
        reply_lines.append("サポート方法: " + "、".join(plan["support"]))
    if analysis.get("referral"):
        reply_lines.append("連絡推奨: " + analysis["referral"])
    if not reply_lines:
        reply_lines.append("ここにいて大丈夫です。どのような夢だったか一緒に整理しましょう。")
    return "\n".join(reply_lines)


def launch_chat():
    with gr.Blocks(title="CBTチャットモック") as demo:
        gr.Markdown("""# CBTチャットモック
夢や気分について入力すると、CBTモックが簡単な提案を返します。""")
        chatbot = gr.Chatbot(label="対話ログ", height=400, type="messages")
        textbox = gr.Textbox(label="メッセージ", placeholder="ここに気持ちや夢の内容を入力してください。", lines=4)
        send_btn = gr.Button("送信")
        clear_btn = gr.Button("リセット")

        def respond(user_input, chat_history):
            if chat_history is None:
                chat_history = []
            chat_history.append({"role": "user", "content": user_input})
            reply = build_reply(user_input)
            chat_history.append({"role": "assistant", "content": reply})
            return "", chat_history

        send_btn.click(respond, inputs=[textbox, chatbot], outputs=[textbox, chatbot])
        textbox.submit(respond, inputs=[textbox, chatbot], outputs=[textbox, chatbot])
        clear_btn.click(lambda: ([], ""), None, outputs=[chatbot, textbox])

    port_env = os.environ.get("GRADIO_SERVER_PORT") or os.environ.get("CHAT_PORT")
    port = int(port_env) if port_env else None
    demo.launch(server_name="0.0.0.0", server_port=port, inbrowser=True)


if __name__ == "__main__":
    launch_chat()
