"""夢を肯定的にリフレーミングし、幸せな記憶とつないだ動画/音声を生成するモックアプリ."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4

import os
import gradio as gr
import matplotlib.pyplot as plt
from gtts import gTTS
from scipy.io import wavfile
import numpy as np
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from dotenv import load_dotenv
from openai import OpenAI

from demo_mini_llm_cbt import FineTunedCBTSystem

ARTIFACT_DIR = Path("artifacts/mock_media")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

for font_name in ("Hiragino Sans", "IPAexGothic", "MS Gothic", "Noto Sans CJK JP", "Noto Sans Japanese"):
    if font_name in plt.rcParams.get("font.family", []):
        break
    try:
        plt.rcParams["font.family"] = font_name
        break
    except Exception:
        continue

CBT_SYSTEM = FineTunedCBTSystem()
load_dotenv()


def get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


OPENAI_CLIENT = get_openai_client()


def wrap_text(text: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(text, width=width)) or text


def create_slide_image(text: str, path: Path) -> None:
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.axis("off")
    plt.text(0.5, 0.5, wrap_text(text), ha="center", va="center", fontsize=18)
    plt.savefig(path, bbox_inches=None, transparent=False)
    plt.close()


def synthesize_media(narration: str, segments: List[str]) -> Tuple[Path, Path]:
    audio_path = ARTIFACT_DIR / f"audio_{uuid4().hex}.mp3"
    try:
        tts = gTTS(narration, lang="ja")
        tts.save(audio_path)
    except Exception:
        duration = 8
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration), False)
        tone = 0.2 * np.sin(2 * np.pi * 220 * t)
        audio_path = audio_path.with_suffix(".wav")
        wavfile.write(audio_path, sr, (tone * 32767).astype(np.int16))

    audio_clip = AudioFileClip(str(audio_path))
    duration = max(audio_clip.duration, 4)
    per_segment = duration / max(len(segments), 1)

    temp_images: List[Path] = []
    for seg in segments:
        img_path = ARTIFACT_DIR / f"slide_{uuid4().hex}.png"
        create_slide_image(seg, img_path)
        temp_images.append(img_path)
    video = ImageSequenceClip([str(p) for p in temp_images], durations=[per_segment] * len(temp_images))
    video = video.with_audio(audio_clip)
    video_path = ARTIFACT_DIR / f"video_{uuid4().hex}.mp4"
    video.write_videofile(
        str(video_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
    )
    video.close()

    audio_clip.close()
    for img in temp_images:
        img.unlink(missing_ok=True)

    return video_path, audio_path


def build_story(dream_part: str, memory_part: str, plan: dict) -> Tuple[str, List[str]]:
    story_parts = [dream_part.strip(), "続けて" + memory_part.strip()]
    if plan.get("primary"):
        story_parts.append("推奨ワーク: " + "、".join(plan["primary"]))
    if plan.get("support"):
        story_parts.append("サポート方法: " + "、".join(plan["support"]))
    narration = "。".join(story_parts)

    slides: List[str] = [dream_part.strip(), memory_part.strip()]
    if plan.get("primary"):
        slides.append("推奨ワーク\n" + "、".join(plan["primary"]))
    if plan.get("support"):
        slides.append("サポート\n" + "、".join(plan["support"]))
    return narration, slides


def call_gpt_reframe(dream: str, memory: str) -> Tuple[str, str] | None:
    if OPENAI_CLIENT is None:
        return None
    prompt = f"""あなたは悪夢ケア専門の臨床心理士です。以下の悪夢を肯定的に再解釈し、続いて幸せな記憶へ自然に移行する物語を考えてください。

悪夢: {dream}
幸せな記憶: {memory}

出力フォーマット:
### 夜のパート
肯定的解釈
### 幸せのパート
幸せな記憶と結び付ける表現
"""
    try:
        response = OPENAI_CLIENT.responses.create(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=400,
        )
        text = response.output_text
    except Exception:
        return None

    night_part = []
    happy_part = []
    current = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "夜のパート" in line:
            current = night_part
            continue
        if "幸せのパート" in line:
            current = happy_part
            continue
        if current is not None:
            current.append(line)
    return ("".join(night_part), "".join(happy_part))


def run_pipeline(dream_text: str, happy_memory: str):
    if not dream_text.strip() or not happy_memory.strip():
        return "夢と幸せな記憶の両方を入力してください。", "", None, None

    analysis = CBT_SYSTEM.process_dream(dream_text.strip())

    gpt_result = call_gpt_reframe(dream_text.strip(), happy_memory.strip())
    if gpt_result:
        dream_reframe, memory_reframe = gpt_result
    else:
        dream_reframe = analysis["response"].strip()
        memory_reframe = happy_memory.strip()

    narration, slides = build_story(dream_reframe, memory_reframe, analysis.get("cbt_plan", {}))
    video_path, audio_path = synthesize_media(narration, slides)

    plan_lines = []
    plan = analysis.get("cbt_plan", {})
    if plan.get("primary"):
        plan_lines.append("【主なワーク】" + "、".join(plan["primary"]))
    if plan.get("support"):
        plan_lines.append("【サポート】" + "、".join(plan["support"]))
    if plan.get("safety"):
        plan_lines.append("【安全策】" + "、".join(plan["safety"]))

    plan_text = "\n".join(plan_lines)
    return narration, plan_text, str(video_path), str(audio_path)


def launch_app():
    with gr.Blocks(title="夢リフレーム動画モック") as demo:
        gr.Markdown("""# 夢リフレーム動画モック
悪夢の内容と幸せな記憶を入力すると、肯定的なストーリーを生成し動画と音声を作成します。""")
        dream_input = gr.Textbox(label="悪夢の内容", placeholder="最近見た夢を具体的に入力してください。", lines=6)
        memory_input = gr.Textbox(label="幸せな記憶", placeholder="安心できた思い出や感謝した出来事を入力してください。", lines=4)
        convert_btn = gr.Button("動画と音声を生成")
        reframed_output = gr.Textbox(label="リフレーミング結果", interactive=False, lines=6)
        plan_output = gr.Textbox(label="提案されたワーク", interactive=False, lines=4)
        video_output = gr.Video(label="生成されたモック動画")
        audio_output = gr.Audio(label="生成された音声", type="filepath")

        convert_btn.click(
            run_pipeline,
            inputs=[dream_input, memory_input],
            outputs=[reframed_output, plan_output, video_output, audio_output],
        )

    port_env = os.environ.get("GRADIO_SERVER_PORT") or os.environ.get("VIDEO_PORT")
    port = int(port_env) if port_env else None
    demo.launch(server_name="0.0.0.0", server_port=port, inbrowser=True)


if __name__ == "__main__":
    launch_app()
