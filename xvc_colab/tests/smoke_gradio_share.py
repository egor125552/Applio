#!/usr/bin/env python3
"""One-shot smoke test for Gradio's public share tunnel.

This intentionally calls demo.launch() exactly once. If a public gradio.live
URL is created, the test prints it and immediately closes the demo.
"""

from __future__ import annotations

import gradio as gr


def main() -> None:
    print(f"Gradio version: {gr.__version__}")
    print("Creating exactly one public Gradio share link...")

    demo = gr.Interface(lambda text: text, gr.Textbox(), gr.Textbox())

    try:
        result = demo.launch(
            share=True,
            prevent_thread_lock=True,
            server_name="127.0.0.1",
            show_error=True,
            quiet=False,
        )

        share_url = getattr(demo, "share_url", None)
        if not share_url and isinstance(result, tuple) and len(result) >= 3:
            share_url = result[2]

        print(f"Share URL returned by Gradio: {share_url!r}")
        if not share_url or not str(share_url).startswith("https://"):
            raise RuntimeError("Gradio launch completed without a public HTTPS share URL")
        if ".gradio.live" not in str(share_url):
            raise RuntimeError(f"Unexpected Gradio share URL: {share_url}")

        print(f"PUBLIC_SHARE_OK: {share_url}")
    finally:
        demo.close()
        print("Gradio demo closed; no second share-link attempt was made.")


if __name__ == "__main__":
    main()
