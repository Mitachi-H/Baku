"""
DiaCBT・Cactusなどの公開ワークブックに基づくCBT知識と、
DreamBankの悪夢記録、DDNSI重症度指標を参照する想定で、
簡易な感情分析・重症度推定・専門家連携判断までを一貫して確認できる。
実際の臨床運用ではここをFineTune済みモデルと実データソースに置き換える。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class KnowledgeDocument:
    """ナレッジベース1件分"""

    doc_id: str
    source: str
    title: str
    technique: str
    category: str
    text: str
    tags: List[str]


@dataclass
class RetrievalResult:
    """RAG検索結果"""

    document: KnowledgeDocument
    score: float


class KnowledgeBase:
    """TF-IDFで簡易検索できるCBT+悪夢知識ベース"""

    def __init__(self) -> None:
        self._documents: List[KnowledgeDocument] = []
        self._vectorizer = TfidfVectorizer(max_features=1200)
        self._matrix = None
        self._seed_documents()

    def _seed_documents(self) -> None:
        """DiaCBT / Cactus / DreamBankを想定したテキストを登録"""

        snippets: Sequence[Dict[str, str | Sequence[str]]] = [
            {
                "source": "DiaCBT",
                "title": "自動思考の棚卸しワーク",
                "technique": "cognitive_restructuring",
                "category": "cbt_core",
                "text": (
                    "出来事・自動思考・感情を記録し、根拠を検証して別の視点を見つける。"
                    "破局的思考を書き直す練習を日々続けることが推奨される。"
                ),
                "tags": ["認知再構成", "思考記録", "DiaCBT"],
            },
            {
                "source": "DiaCBT",
                "title": "マインドフルネス呼吸誘導",
                "technique": "mindfulness",
                "category": "mindfulness",
                "text": (
                    "4-6カウントの腹式呼吸と五感への注意で現在地に戻る。"
                    "呼吸後に感謝メモを書くと情動調整が安定する。"
                ),
                "tags": ["呼吸", "気づき", "セルフコンパッション"],
            },
            {
                "source": "Cactus",
                "title": "行動活性化ミニ課題リスト",
                "technique": "behavioral_activation",
                "category": "cbt_support",
                "text": (
                    "楽しみと達成感の小さな行動をスケジュール化。実施前後で気分を評価し、"
                    "週間レビューで習慣化を確認する。"
                ),
                "tags": ["行動活性化", "スケジューリング", "行動実験"],
            },
            {
                "source": "DreamBank",
                "title": "イメージリハーサル基本プロトコル",
                "technique": "imagery_rehearsal",
                "category": "nightmare_protocol",
                "text": (
                    "悪夢を安全な結末へ書き換え、五感を使って1日2回リハーサルする。"
                    "悪夢日誌に情動の変化と自信の推移を記録する。"
                ),
                "tags": ["悪夢", "書き換え", "リハーサル"],
            },
            {
                "source": "DreamBank",
                "title": "DDNSI評価メモ",
                "technique": "severity_monitoring",
                "category": "assessment",
                "text": (
                    "頻度・覚醒・苦痛度・翌日の不安を定量化し、スコア合計で重症度帯を判定する。"
                    "睡眠専門医への紹介判断材料として活用する。"
                ),
                "tags": ["DDNSI", "重症度", "モニタリング"],
            },
            {
                "source": "臨床ガイド",
                "title": "専門医紹介チェックリスト",
                "technique": "specialist_support",
                "category": "safety",
                "text": (
                    "DDNSIが15点以上、日中機能の低下、自傷念慮が見られる場合は特定専門医に連絡。"
                    "地域の睡眠外来・危機対応窓口リストを共有する。"
                ),
                "tags": ["紹介", "安全管理", "連携"],
            },
        ]

        for idx, snippet in enumerate(snippets):
            self._documents.append(
                KnowledgeDocument(
                    doc_id=f"kb_{idx:03d}",
                    source=str(snippet["source"]),
                    title=str(snippet["title"]),
                    technique=str(snippet["technique"]),
                    category=str(snippet["category"]),
                    text=str(snippet["text"]),
                    tags=list(snippet["tags"]),
                )
            )

        self._matrix = self._vectorizer.fit_transform([doc.text for doc in self._documents])

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        if not query.strip():
            return []
        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix)[0]
        order = np.argsort(sims)[::-1][:top_k]
        return [RetrievalResult(self._documents[i], float(sims[i])) for i in order if sims[i] > 1e-3]

    @property
    def documents(self) -> Sequence[KnowledgeDocument]:
        return self._documents


class EmotionAnalyzer:
    """簡易的な感情・原感情推定"""

    NEGATIVE_TERMS = {
        "怖い": 0.9,
        "恐ろしい": 1.0,
        "不安": 0.7,
        "焦り": 0.6,
        "絶望": 0.9,
        "怒り": 0.6,
        "孤独": 0.5,
        "悲しい": 0.6,
    }
    POSITIVE_TERMS = {
        "安心": 0.7,
        "穏やか": 0.6,
        "感謝": 0.5,
        "希望": 0.6,
        "支えられた": 0.5,
        "落ち着いた": 0.5,
    }

    def analyse(self, text: str) -> Dict[str, float | str]:
        lowered = text.lower()
        neg = sum(weight for token, weight in self.NEGATIVE_TERMS.items() if token in lowered)
        pos = sum(weight for token, weight in self.POSITIVE_TERMS.items() if token in lowered)
        raw_score = pos - neg
        if raw_score > 0.5:
            label = "ポジティブ"
        elif raw_score < -0.5:
            label = "強いストレス"
        else:
            label = "混在"
        intensity = min(1.0, abs(raw_score) / 3 + 0.2 * (neg > 0))
        primary = "恐怖" if any(word in lowered for word in ("怖", "恐ろしい", "震え")) else label
        return {
            "label": label,
            "intensity": round(float(intensity), 3),
            "primary_emotion": primary,
            "valence": round(float(raw_score), 3),
        }


class NightmareSeverityScorer:
    """DDNSIを模した簡易スコアリング"""

    DDNSI_WEIGHTS = {
        "悪夢": 4,
        "飛び起き": 3,
        "叫ん": 2,
        "入眠困難": 2,
        "回避": 2,
        "トラウマ": 3,
        "暴力": 2,
        "血": 2,
        "死": 3,
    }

    def score(self, text: str, emotion: Dict[str, float | str]) -> Dict[str, float | str]:
        lowered = text.lower()
        base = 4.0
        for token, weight in self.DDNSI_WEIGHTS.items():
            if token in lowered:
                base += weight
        base += 2 if emotion.get("label") == "強いストレス" else 0
        base = min(base, 20.0)
        band = "重度" if base >= 15 else "中等度" if base >= 10 else "軽度"
        return {
            "ddnsi": round(base, 2),
            "severity_band": band,
            "needs_escalation": band == "重度",
        }


class CBTResponseGenerator:
    """FineTune済みLLMを模したテンプレート応答"""

    TEMPLATES: Dict[str, Sequence[str]] = {
        "cognitive_restructuring": (
            "浮かんだ自動思考を書き出し、証拠を丁寧に振り返ってみましょう。"
            "客観的な視点や慈悲的な言葉に置き換える練習を今夜行ってみてください。"
        ),
        "imagery_rehearsal": (
            "悪夢の終わり方を安全で力強い展開に書き換え、全身でその場面を感じながら"
            "寝る前と日中にリハーサルしましょう。"
        ),
        "mindfulness": (
            "ゆっくりとした腹式呼吸と5-4-3-2-1グラウンディングで、今この瞬間に注意を戻しましょう。"
        ),
        "behavioral_activation": (
            "翌朝の達成行動と夕方の楽しみ行動を1つずつ予定に入れ、気分の変化をメモしてみましょう。"
        ),
        "specialist_support": (
            "重症度が上がっています。信頼できる睡眠専門医に連絡し、記録を共有できるよう準備しましょう。"
        ),
    }

    def generate(self, technique: str, context: Sequence[RetrievalResult], emotion: Dict[str, float | str]) -> str:
        template = self.TEMPLATES.get(
            technique,
            "今夜は安全を感じられるサポートを整え、どんなサポートが役立つか一緒に探っていきましょう。",
        )
        if context:
            citations = "、".join(f"{item.document.source}:{item.document.title}" for item in context)
            template += f"\n参考: {citations}"
        if emotion.get("label") == "強いストレス":
            template += "\n強い感情が出たら夢日記にメモし、次回の対話で共有してください。"
        return template


class SpecialistReferralEngine:
    """専門家紹介が必要か判断"""

    def decide(self, severity: Dict[str, float | str], emotion: Dict[str, float | str]) -> Optional[str]:
        if severity.get("needs_escalation"):
            return "重度指標が確認されました。睡眠専門医と安全計画について相談しましょう。"
        if emotion.get("primary_emotion") == "恐怖" and severity.get("ddnsi", 0) >= 12:
            return "恐怖感が強い状態です。オンラインの専門医相談を提案してください。"
        return None


class FineTunedCBTSystem:
    """RAGと感情分析を組み合わせた一連の処理"""

    def __init__(self) -> None:
        self.knowledge_base = KnowledgeBase()
        self.emotion_analyzer = EmotionAnalyzer()
        self.severity_scorer = NightmareSeverityScorer()
        self.response_generator = CBTResponseGenerator()
        self.referral_engine = SpecialistReferralEngine()
        self.session_history: List[Dict[str, object]] = []
        self._fine_tune_recipe = self._build_finetune_recipe()

    def _build_finetune_recipe(self) -> Dict[str, object]:
        return {
            "datasets": [
                {"name": "DiaCBT", "role": "認知再構成の対話パターン"},
                {"name": "Cactus", "role": "行動活性化と宿題設計の例"},
                {"name": "DreamBank", "role": "悪夢ナラティブと書き換え事例"},
                {"name": "DDNSI", "role": "重症度ラベルによる安全判定"},
            ],
            "alignment": {
                "technique_head": ["cognitive_restructuring", "imagery_rehearsal", "mindfulness"],
                "safety_classification": "DDNSI帯域 + 危機関連語彙",
                "evaluation": "専門家レビューによる継続評価",
            },
        }

    def get_finetune_plan(self) -> Dict[str, object]:
        return self._fine_tune_recipe

    def _choose_technique(self, results: Sequence[RetrievalResult], severity: Dict[str, float | str]) -> str:
        if severity.get("ddnsi", 0) >= 15:
            return "specialist_support"
        if not results:
            return "mindfulness"
        ranked = sorted(results, key=lambda item: item.score, reverse=True)
        for item in ranked:
            if item.document.technique not in {"severity_monitoring", "specialist_support"}:
                return item.document.technique
        return ranked[0].document.technique

    def _confidence(self, results: Sequence[RetrievalResult], severity: Dict[str, float | str]) -> float:
        base = 0.55 + 0.1 * min(len(results), 3)
        adjust = -0.1 if severity.get("severity_band") == "重度" else 0.05
        return float(min(0.95, max(0.4, base + adjust)))

    def _craft_plan(self, technique: str) -> Dict[str, Iterable[str]]:
        mindfulness_support = ["寝る前3分のボックス呼吸", "5-4-3-2-1グラウンディング"]
        core = {
            "cognitive_restructuring": ["思考記録シート", "証拠リストの記入"],
            "imagery_rehearsal": ["悪夢の書き換え台本", "日々のリハーサル記録"],
            "mindfulness": ["ボディスキャン音声", "慈悲の言葉リスト"],
            "behavioral_activation": ["行動スケジュール表", "気分チェック表"],
            "specialist_support": ["睡眠専門医との共有資料作成", "安全確保プラン"],
        }
        return {
            "primary": core.get(technique, ["夢日記の継続"]),
            "support": mindfulness_support,
            "safety": ["感情が高ぶったらセーフティプランを確認", "睡眠衛生チェックリスト"],
        }

    def process_dream(self, dream_text: str, user_profile: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        emotion = self.emotion_analyzer.analyse(dream_text)
        severity = self.severity_scorer.score(dream_text, emotion)
        retrieval = self.knowledge_base.retrieve(dream_text, top_k=3)
        technique = self._choose_technique(retrieval, severity)
        response = self.response_generator.generate(technique, retrieval, emotion)
        confidence = self._confidence(retrieval, severity)
        referral = self.referral_engine.decide(severity, emotion)
        plan = self._craft_plan(technique)

        result = {
            "technique": technique,
            "response": response,
            "confidence": round(confidence, 3),
            "emotion": emotion,
            "severity": severity,
            "rag_sources": [
                {
                    "doc_id": item.document.doc_id,
                    "title": item.document.title,
                    "source": item.document.source,
                    "score": round(item.score, 3),
                }
                for item in retrieval
            ],
            "cbt_plan": plan,
            "finetune_plan": self.get_finetune_plan(),
            "referral": referral,
        }

        self.session_history.append(
            {
                "input": dream_text,
                "analysis": result,
                "user_profile": user_profile or {},
            }
        )

        return result


__all__ = [
    "FineTunedCBTSystem",
    "KnowledgeBase",
    "KnowledgeDocument",
    "RetrievalResult",
]
