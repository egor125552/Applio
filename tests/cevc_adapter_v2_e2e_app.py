"""Launch only the CEVC 2B tab for browser end-to-end validation."""

from __future__ import annotations

import os

import gradio as gr

from tabs.cevc2b.lab import cevc2b_lab_tab


PORT = int(os.environ.get("CEVC_E2E_PORT", "7867"))

with gr.Blocks(title="CEVC Adapter v2 E2E") as demo:
    cevc2b_lab_tab()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=PORT,
        share=False,
        show_error=True,
        prevent_thread_lock=False,
    )
