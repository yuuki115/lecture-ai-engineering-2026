"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --interactive

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    print_usage(response)
    return response.text


def print_usage(response) -> None:
    """レスポンスに含まれるトークン使用量を表示する。"""
    usage = response.usage_metadata
    print(f"[入力トークン数] {usage.prompt_token_count}")
    print(f"[出力トークン数] {usage.candidates_token_count}")


def ask_interactively(initial_question: str | None = None) -> None:
    """1つの Chat セッションで、履歴を保ったまま質問を繰り返す。"""
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)

    print("対話モードです。終了するには quit、exit、または Ctrl-D を入力してください。")
    question = initial_question
    while True:
        if question is None:
            try:
                question = input("\n質問> ").strip()
            except EOFError:
                print()
                return
        if question.lower() in {"quit", "exit"}:
            return
        if not question:
            question = None
            continue

        response = chat.send_message(question)
        print_usage(response)
        print(f"\n{response.text}")
        question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="会話履歴を保ったまま、続けて質問する",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.interactive:
            ask_interactively(args.question)
        else:
            question = args.question or DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
