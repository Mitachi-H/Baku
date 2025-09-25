"""REM解析とCBTモックを結合したデモ（日本語版）。"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from demo_mini_llm_cbt import FineTunedCBTSystem


class DreamTherapyMockApp:
    """REM指標・感情分析・RAG応答をひとまとめに扱う統合モック"""

    def __init__(self) -> None:
        self.cbt_system = FineTunedCBTSystem()
        self.sessions: Dict[str, Dict[str, object]] = {}

    def ingest_sleep_metrics(
        self,
        session_id: str,
        rem_percentage: float,
        rem_episodes: int,
        total_minutes: float,
        dream_report: Optional[str] = None,
        speech_sentiment: Optional[str] = None,
    ) -> None:
        """睡眠指標を保存し、夢レポートがあればCBT解析を実行"""

        record = {
            "timestamp": datetime.now(),
            "rem_percentage": rem_percentage,
            "rem_episodes": rem_episodes,
            "total_minutes": total_minutes,
            "dream_report": dream_report,
            "speech_sentiment": speech_sentiment,
            "analysis": None,
        }

        if dream_report:
            record["analysis"] = self.cbt_system.process_dream(
                dream_report,
                user_profile={"spoken_valence": speech_sentiment},
            )

        self.sessions[session_id] = record

    def build_report(self, session_id: str) -> Dict[str, object]:
        if session_id not in self.sessions:
            raise KeyError(f"未知のセッションID: {session_id}")

        session = self.sessions[session_id]
        rem_quality = "注意" if session["rem_percentage"] < 18 else "良好"
        next_steps: List[str] = ["毎晩の夢日記を続けましょう。"]
        if session["rem_percentage"] < 15:
            next_steps.append("睡眠衛生を見直し、継続するなら専門家と相談してください。")
        if session["analysis"] and session["analysis"].get("referral"):
            next_steps.append("専門医連携の案内に24時間以内に返信しましょう。")
        if session["speech_sentiment"] == "ネガティブ":
            next_steps.append("マインドフルネス音声ガイドを案内し、安心できる環境を整えましょう。")

        return {
            "session_id": session_id,
            "timestamp": session["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "sleep_metrics": {
                "rem_percentage": session["rem_percentage"],
                "rem_episodes": session["rem_episodes"],
                "total_minutes": session["total_minutes"],
                "rem_quality": rem_quality,
            },
            "dream_analysis": session["analysis"],
            "finetune_plan": self.cbt_system.get_finetune_plan(),
            "pro_recommendations": next_steps,
        }


def demo() -> None:
    """モックアプリの挙動をサンプル入力で表示"""

    app = DreamTherapyMockApp()

    app.ingest_sleep_metrics(
        session_id="night_001",
        rem_percentage=16.2,
        rem_episodes=3,
        total_minutes=420,
        dream_report="暗い廊下で何かに追いかけられ、震えながら目が覚めました。もう寝るのが怖いです。",
        speech_sentiment="ネガティブ",
    )

    app.ingest_sleep_metrics(
        session_id="night_002",
        rem_percentage=22.4,
        rem_episodes=5,
        total_minutes=445,
        dream_report="海の上を滑空し、友人に支えられて安心した気持ちで目覚めました。",
        speech_sentiment="ポジティブ",
    )

    for session_id in ("night_001", "night_002"):
        report = app.build_report(session_id)
        print("=" * 68)
        print(f"セッション: {report['session_id']} ({report['timestamp']})")
        metrics = report["sleep_metrics"]
        print(
            f"REM割合: {metrics['rem_percentage']:.1f}% / エピソード: {metrics['rem_episodes']}回 / "
            f"総睡眠時間: {metrics['total_minutes']}分 / 評価: {metrics['rem_quality']}"
        )

        if report["dream_analysis"]:
            analysis = report["dream_analysis"]
            print("\n-- LLMコーチングサマリー --")
            print(f"推奨技法: {analysis['technique']} (信頼度 {analysis['confidence']})")
            print(f"感情推定: {analysis['emotion']}")
            print(f"重症度推定: {analysis['severity']}")
            print(f"推奨プラン: {analysis['cbt_plan']}")
            if analysis.get("referral"):
                print(f"専門家連携: {analysis['referral']}")
            print("参考文献:")
            for src in analysis["rag_sources"]:
                print(f"  - {src['source']} / {src['title']} (score={src['score']})")
        else:
            print("夢の記録がありません。次回は出来事のメモをお願いしましょう。")

        print("\n推奨アクション:")
        for item in report["pro_recommendations"]:
            print(f"  * {item}")

    print("=" * 68)


if __name__ == "__main__":
    demo()
