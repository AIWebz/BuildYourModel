"""Chat with Basis in the terminal.  Usage: python chat.py [--debug]"""
import sys

from ai import Assistant


def main():
    debug = "--debug" in sys.argv
    bot = Assistant().load(verbose=True)
    print("Basis is ready. Type 'quit' to exit. Commands: /teach question | answer, /generate words\n")
    while True:
        try:
            text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() in {"quit", "exit"}:
            break
        if debug:
            for label, score, p, s in bot.analyze(text):
                print(f"   [brain] {label:14} score {score:.2f} (net {p:.2f}, retrieval {s:.2f})")
        print("Basis:", bot.reply(text))


if __name__ == "__main__":
    main()
