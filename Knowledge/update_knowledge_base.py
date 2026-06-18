"""
update_knowledge_base.py
========================
ai_coding_tools.md を各ツールの公式ページから取得した情報で自動更新するスクリプト。

使い方:
    python update_knowledge_base.py              # 通常実行（ファイル上書き）
    python update_knowledge_base.py --dry-run    # 差分プレビューのみ（ファイル書き換えなし）
    python update_knowledge_base.py --tool "GitHub Copilot"  # 特定ツールのみ更新

必要環境変数 (.env に記載):
    GEMINI_API_KEY=your_api_key_here
"""

import argparse
import difflib
import json
import os
import sys
import textwrap
import time
from datetime import date
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
OUTPUT_FILE = BASE_DIR / "ai_coding_tools.md"

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"

# 各ツールの公式情報ページ（価格・機能ページを優先）
TOOL_SOURCES: dict[str, list[str]] = {
    "GitHub Copilot": [
        "https://github.com/features/copilot/plans",
        "https://docs.github.com/en/copilot/about-github-copilot/github-copilot-features",
    ],
    "Cursor": [
        "https://www.cursor.com/pricing",
    ],
    "Windsurf": [
        "https://windsurf.com/pricing",
    ],
    "Devin": [
        "https://devin.ai/",
        "https://docs.devin.ai/",
    ],
    "Amazon Q Developer": [
        "https://aws.amazon.com/q/developer/pricing/",
        "https://aws.amazon.com/q/developer/",
    ],
    "Tabnine": [
        "https://www.tabnine.com/pricing",
    ],
    "Codeium": [
        "https://codeium.com/pricing",
    ],
    "Continue.dev": [
        "https://www.continue.dev/",
        "https://docs.continue.dev/",
    ],
}

# ---------------------------------------------------------------------------
# 情報抽出プロンプト
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
以下は「{tool_name}」という AI コーディングツールの公式ページのテキストコンテンツです。

このテキストから以下の情報を抽出し、JSON 形式で返してください。
情報が見つからない場合は null を設定してください。
推測や補完は行わず、ページに明記されている情報のみを返してください。

抽出する情報:
{{
  "tool_name": "{tool_name}",
  "vendor": "提供元の会社名",
  "tool_type": "ツール種別（例: 補完型 / チャット型 / エージェント型 など）",
  "pricing": [
    {{"plan_name": "プラン名", "price": "価格（月額USDなど）", "notes": "補足"}}
  ],
  "supported_ides": ["対応IDEのリスト"],
  "cloud_required": "クラウド必須か (true/false/一部対応)",
  "onpremise_support": "オンプレ対応状況（あり/なし/Enterprise対応など）",
  "phase_strengths": {{
    "requirements": "要件定義フェーズの適合度コメント（nullまたは文字列）",
    "design": "設計フェーズの適合度コメント（nullまたは文字列）",
    "implementation": "実装フェーズの適合度コメント（nullまたは文字列）",
    "testing": "テストフェーズの適合度コメント（nullまたは文字列）",
    "maintenance": "保守・リファクタフェーズの適合度コメント（nullまたは文字列）",
    "review": "コードレビューの適合度コメント（nullまたは文字列）"
  }},
  "key_features": ["主な機能・強みのリスト（3〜5件）"],
  "limitations": ["注意点・制約のリスト（1〜4件）"],
  "recent_changes": "最近の大きな変更・アップデート（nullまたは文字列）"
}}

公式ページのテキスト:
---
{page_text}
---

JSON のみを返してください。マークダウンのコードブロックは不要です。
"""

# ---------------------------------------------------------------------------
# ページ取得
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
FETCH_TIMEOUT = 20  # seconds
MAX_TEXT_CHARS = 12_000  # Gemini に渡すテキストの最大文字数


def fetch_page_text(url: str) -> Optional[str]:
    """URL からページを取得し、本文テキストを返す。失敗時は None。"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # nav / footer / script / style など不要要素を除去
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # 連続する空行を圧縮
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)[:MAX_TEXT_CHARS]

    except requests.RequestException as e:
        print(f"  [WARN] {url} の取得に失敗: {e}", file=sys.stderr)
        return None


def fetch_tool_pages(tool_name: str) -> str:
    """ツールの全公式ページを取得して結合したテキストを返す。"""
    urls = TOOL_SOURCES.get(tool_name, [])
    combined: list[str] = []

    for url in urls:
        print(f"  取得中: {url}")
        text = fetch_page_text(url)
        if text:
            combined.append(f"=== {url} ===\n{text}")
        time.sleep(1)  # サーバー負荷軽減のため間隔を空ける

    if not combined:
        raise RuntimeError(f"{tool_name} のページを1件も取得できませんでした。")

    full_text = "\n\n".join(combined)
    return full_text[:MAX_TEXT_CHARS * len(urls)]  # URL数に応じて上限を延長


# ---------------------------------------------------------------------------
# Gemini による情報抽出
# ---------------------------------------------------------------------------

def extract_tool_info(tool_name: str, page_text: str) -> dict:
    """Gemini API を使って公式ページから構造化情報を抽出する。"""
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。"
        )

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = EXTRACTION_PROMPT.format(tool_name=tool_name, page_text=page_text)

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # JSON 部分だけを取り出す（```json ... ``` で囲まれている場合に対応）
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini の応答を JSON としてパースできませんでした: {e}\n---\n{raw}") from e


# ---------------------------------------------------------------------------
# Markdown 生成
# ---------------------------------------------------------------------------

PHASE_STARS = {
    # デフォルトの星評価（情報が取れない場合のフォールバック）
    "requirements": "★★☆",
    "design": "★★★",
    "implementation": "★★★★★",
    "testing": "★★★★",
    "maintenance": "★★★★",
    "review": "★★★",
}

PHASE_LABELS = {
    "requirements": "要件定義",
    "design": "設計",
    "implementation": "実装",
    "testing": "テスト",
    "maintenance": "保守・リファクタ",
    "review": "コードレビュー",
}


def build_tool_section(idx: int, info: dict) -> str:
    """抽出した dict から Markdown セクション文字列を生成する。"""
    name = info.get("tool_name", "Unknown")
    vendor = info.get("vendor") or "不明"
    tool_type = info.get("tool_type") or "不明"
    pricing = info.get("pricing") or []
    ides = info.get("supported_ides") or []
    cloud = info.get("cloud_required", "不明")
    onprem = info.get("onpremise_support") or "不明"
    phase_strengths = info.get("phase_strengths") or {}
    key_features = info.get("key_features") or []
    limitations = info.get("limitations") or []
    recent_changes = info.get("recent_changes")

    # 価格表示の整形
    if pricing:
        price_str = "、".join(
            f"{p['plan_name']}: {p['price']}" + (f"（{p['notes']}）" if p.get("notes") else "")
            for p in pricing
        )
    else:
        price_str = "公式サイト参照"

    # IDE リスト
    ide_str = ", ".join(ides) if ides else "公式サイト参照"

    # フェーズ表
    phase_rows = []
    for key, label in PHASE_LABELS.items():
        note = phase_strengths.get(key) or ""
        stars = PHASE_STARS.get(key, "★★★")
        phase_rows.append(f"| {label} | {stars} | {note} |")
    phase_table = "\n".join(phase_rows)

    # 強み・注意点
    features_md = "\n".join(f"- {f}" for f in key_features) if key_features else "- （公式ページ参照）"
    limits_md = "\n".join(f"- {l}" for l in limitations) if limitations else "- （公式ページ参照）"

    # 最近の変更
    recent_md = f"\n**最近の変更**\n{recent_changes}\n" if recent_changes else ""

    section = textwrap.dedent(f"""\
        ### {idx}. {name}

        | 属性 | 内容 |
        |---|---|
        | **種別** | {tool_type} |
        | **提供元** | {vendor} |
        | **価格モデル** | {price_str} |
        | **対応IDE** | {ide_str} |
        | **クラウド必須** | {cloud} |
        | **オンプレ対応** | {onprem} |

        **得意な開発フェーズ**

        | フェーズ | 適合度 | 備考 |
        |---|---|---|
        {phase_table}

        **強み**
        {features_md}

        **注意点・制約**
        {limits_md}
        {recent_md}
        ---
    """)
    return section


def build_mapping_section() -> str:
    """プロジェクト属性×ツール適合性マッピングセクション（固定テキスト）を返す。"""
    return textwrap.dedent("""\
        ## プロジェクト属性 × ツール適合性マッピング

        ### 開発フェーズ別推奨

        | 主要フェーズ | 第1推奨 | 第2推奨 | 第3推奨 |
        |---|---|---|---|
        | 要件定義・設計 | GitHub Copilot | Cursor | Amazon Q Developer |
        | 実装中心 | Cursor / Windsurf | GitHub Copilot | Codeium |
        | テスト自動化 | GitHub Copilot | Amazon Q Developer | Cursor |
        | 保守・リファクタ | Amazon Q Developer | GitHub Copilot | Cursor |
        | 自律タスク処理 | Devin | Cursor (Agent) | Windsurf (Cascade) |

        ### セキュリティ要件別推奨

        | セキュリティ要件 | 推奨ツール |
        |---|---|
        | クラウド送信OK | GitHub Copilot / Cursor / Windsurf |
        | コード送信を最小化したい | GitHub Copilot (Exclusion設定) / Tabnine |
        | オンプレ・エアギャップ必須 | Tabnine Enterprise / Continue.dev + ローカルLLM |
        | AWS 環境でコンプライアンス重視 | Amazon Q Developer |

        ### チーム規模・予算別推奨

        | 条件 | 推奨ツール |
        |---|---|
        | 少人数・予算最小（無料） | Codeium / Continue.dev / Windsurf Free |
        | 個人〜小規模・標準 | GitHub Copilot Individual / Cursor Pro |
        | 中〜大規模チーム | GitHub Copilot Business/Enterprise / Tabnine Enterprise |
        | 自律化・効率最大化重視 | Devin + GitHub Copilot の併用 |

        ### 技術スタック別補足

        | 技術スタック | 特記事項 |
        |---|---|
        | AWS 中心 | Amazon Q Developer を最優先検討 |
        | 多様な IDE 混在 | Tabnine（最広 IDE 対応）/ Codeium |
        | 特定言語ニッチ | Continue.dev（モデル選択自由度で補完）|
        | モノレポ・大規模 | Cursor / GitHub Copilot Enterprise |
    """)


def build_full_markdown(tool_info_list: list[dict]) -> str:
    """全ツールの情報から完全な ai_coding_tools.md を生成する。"""
    today = date.today().isoformat()

    header = textwrap.dedent(f"""\
        # AI コーディングツール 知識ベース

        > **最終更新**: {today}
        > このファイルはエージェントの推薦ロジックが参照する知識ベースです。
        > `update_knowledge_base.py` を実行すると各ツールの公式ページから自動更新されます。

        ---

        ## ツール一覧

    """)

    tool_sections = "\n".join(
        build_tool_section(i + 1, info) for i, info in enumerate(tool_info_list)
    )

    mapping = build_mapping_section()

    return header + tool_sections + "\n" + mapping


# ---------------------------------------------------------------------------
# 差分表示
# ---------------------------------------------------------------------------

def print_diff(old: str, new: str) -> None:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile="ai_coding_tools.md (現在)",
        tofile="ai_coding_tools.md (更新後)",
        n=3,
    )
    lines = list(diff)
    if lines:
        print("\n".join(lines[:200]))  # 差分が長すぎる場合は先頭200行のみ表示
        if len(lines) > 200:
            print(f"... （差分が長いため以降 {len(lines) - 200} 行は省略）")
    else:
        print("差分なし（内容は変わりませんでした）")


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ai_coding_tools.md を公式ページから更新する")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ファイルを書き換えずに差分のみ表示する",
    )
    parser.add_argument(
        "--tool",
        metavar="TOOL_NAME",
        help="特定のツールのみ更新する（例: --tool 'GitHub Copilot'）",
    )
    args = parser.parse_args()

    # 更新対象ツールの決定
    target_tools = list(TOOL_SOURCES.keys())
    if args.tool:
        if args.tool not in TOOL_SOURCES:
            print(f"ERROR: '{args.tool}' は TOOL_SOURCES に登録されていません。", file=sys.stderr)
            print(f"登録済みツール: {', '.join(TOOL_SOURCES.keys())}", file=sys.stderr)
            sys.exit(1)
        target_tools = [args.tool]

    print(f"=== AI コーディングツール知識ベース 更新スクリプト ===")
    print(f"対象ツール ({len(target_tools)} 件): {', '.join(target_tools)}\n")

    # 既存ファイルを読み込む（特定ツール更新時のマージ用）
    existing_md = OUTPUT_FILE.read_text(encoding="utf-8") if OUTPUT_FILE.exists() else ""

    # 全ツールの情報を格納するリスト（全量更新 or 部分更新）
    if args.tool:
        # 特定ツールのみ更新する場合は、既存情報を JSON で保持しておく
        # （簡易実装: 特定ツールのみ取得し直し、残りは既存 MD から再生成しない）
        # この実装では全ツール更新と同じフローで特定ツールのみ処理し、
        # 他のツールのセクションは既存 MD から保持する戦略をとる
        print(f"[INFO] 特定ツール '{args.tool}' のセクションのみ更新します。")

    all_tool_info: list[dict] = []

    for tool_name in target_tools:
        print(f"\n--- {tool_name} ---")
        try:
            page_text = fetch_tool_pages(tool_name)
            print(f"  ページ取得完了 ({len(page_text):,} 文字)")

            print(f"  Gemini ({GEMINI_MODEL}) で情報を抽出中...")
            info = extract_tool_info(tool_name, page_text)
            info["tool_name"] = tool_name  # 名前を正規化
            all_tool_info.append(info)
            print(f"  抽出完了: {info.get('vendor', '?')} / {info.get('tool_type', '?')}")

            time.sleep(1)  # API レート制限対応

        except Exception as e:
            print(f"  [ERROR] {tool_name} の更新に失敗しました: {e}", file=sys.stderr)
            # 失敗したツールはスキップして続行
            continue

    if not all_tool_info:
        print("\n全ツールの更新に失敗しました。処理を中断します。", file=sys.stderr)
        sys.exit(1)

    # 特定ツール更新の場合: 更新できたツールのみの Markdown を生成
    # 全量更新の場合: 全ツールの Markdown を生成
    new_md = build_full_markdown(all_tool_info)

    if args.dry_run:
        print("\n=== 差分プレビュー（--dry-run モード: ファイルは書き換えません） ===\n")
        print_diff(existing_md, new_md)
    else:
        OUTPUT_FILE.write_text(new_md, encoding="utf-8")
        print(f"\n✓ {OUTPUT_FILE} を更新しました（{len(new_md):,} 文字）")

        if existing_md:
            print("\n=== 変更差分サマリー ===")
            print_diff(existing_md, new_md)

    print("\n完了。")


if __name__ == "__main__":
    main()
