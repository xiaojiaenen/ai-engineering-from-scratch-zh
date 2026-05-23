#!/usr/bin/env python3
"""Translate all lesson docs (en.md -> zh.md) using OpenAI-compatible API.
Preserves code blocks, inline code, and markdown formatting.
Skips already-translated files."""

import json
import re
import ssl
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

API_BASE = "https://token-plan-cn.xiaomimimo.com/v1"
API_KEY = "tp-cbsgysth0q8iuh5e9e81ha9uzxdc4zdrrxfbh7z9gp60zz3e"
MODEL = "mimo-v2.5-pro"

# Unverified SSL context for custom endpoints
SSL_CONTEXT = ssl._create_unverified_context()


def protect_code_blocks(text: str) -> tuple[str, list[str], list[str]]:
    """Replace fenced code blocks and inline code with placeholders."""
    # Collect fenced code blocks (```...```)
    fence_pattern = re.compile(r'```[\s\S]*?```')
    fenced = fence_pattern.findall(text)
    result = text
    for i, block in enumerate(fenced):
        placeholder = f"[CODEBLOCK_{i}]"
        result = result.replace(block, placeholder, 1)

    # Collect inline code (`...`)
    inline_pattern = re.compile(r'`[^`\n]+`')
    inlines = inline_pattern.findall(result)
    for i, code in enumerate(inlines):
        placeholder = f"[INLINE_{i}]"
        result = result.replace(code, placeholder, 1)

    return result, fenced, inlines


def restore_code_blocks(text: str, fenced: list[str], inlines: list[str]) -> str:
    """Restore code blocks from placeholders."""
    result = text
    for i, block in enumerate(fenced):
        result = result.replace(f"[CODEBLOCK_{i}]", block)
    for i, code in enumerate(inlines):
        result = result.replace(f"[INLINE_{i}]", code)
    return result


def call_api(text: str, retries: int = 3) -> Optional[str]:
    """Call the translation API."""
    system_prompt = (
        "You are a technical translator specialized in AI/ML content. "
        "Translate the following English markdown to Simplified Chinese (zh-CN). "
        "Rules:\n"
        "- Preserve ALL markdown formatting exactly (headers, lists, tables, links, bold, italic)\n"
        "- Preserve ALL placeholders like [CODEBLOCK_N] and [INLINE_N] EXACTLY as-is. These represent source code.\n"
        "- Translate technical terms accurately and consistently. Key terms:\n"
        "  'backpropagation' -> '反向传播', 'gradient' -> '梯度', 'loss function' -> '损失函数',\n"
        "  'weight' -> '权重', 'bias' -> '偏置', 'activation' -> '激活', 'layer' -> '层',\n"
        "  'embedding' -> '嵌入', 'attention' -> '注意力', 'token' -> 'token',\n"
        "  'fine-tuning' -> '微调', 'inference' -> '推理', 'training' -> '训练'\n"
        "- Use natural, fluent Chinese suitable for technical education\n"
        "- Translate the meaning, not word-for-word\n"
        "- Keep mermaid diagram descriptions unchanged (the text inside graph nodes)\n"
        "- Do NOT add any extra commentary, notes, or explanations\n"
        "- Output ONLY the translated markdown, nothing else"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,
        "max_tokens": 8192,
    }

    data = json.dumps(payload).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{API_BASE}/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                },
            )
            with urllib.request.urlopen(req, timeout=180, context=SSL_CONTEXT) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                return content
        except Exception as e:
            print(f"    Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    return None


def translate_file(en_path: Path) -> bool:
    """Translate a single en.md file. Returns True on success."""
    zh_path = en_path.parent / "zh.md"

    # Skip if already translated (file > 100 bytes)
    if zh_path.exists() and zh_path.stat().st_size > 100:
        return True

    text = en_path.read_text(encoding="utf-8")
    if not text.strip():
        return True

    # Protect code blocks
    protected, fenced, inlines = protect_code_blocks(text)

    # Translate
    translated = call_api(protected)
    if translated is None:
        return False

    # Restore code blocks
    final = restore_code_blocks(translated, fenced, inlines)

    # Save
    zh_path.write_text(final, encoding="utf-8")
    return True


def main():
    project_root = Path(__file__).resolve().parent.parent
    phase_dirs = sorted(project_root.glob("phases/*/"))
    all_files = []
    for phase_dir in phase_dirs:
        for en_file in sorted(phase_dir.rglob("docs/en.md")):
            all_files.append(en_file)

    total = len(all_files)
    done = 0
    failed = 0
    skipped = 0
    failed_files = []

    print(f"Found {total} lesson documents across {len(phase_dirs)} phases")
    print(f"API: {API_BASE} | Model: {MODEL}")
    print(f"Skipping already-translated files (zh.md exists and > 100 bytes)")
    print("=" * 60)

    start_time = time.time()

    for i, en_file in enumerate(all_files):
        zh_file = en_file.parent / "zh.md"
        if zh_file.exists() and zh_file.stat().st_size > 100:
            skipped += 1
            if (i + 1) % 20 == 0 or i == total - 1:
                elapsed = time.time() - start_time
                rate = (done + skipped) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{total}] Progress: {done} done, {skipped} skipped, "
                      f"{failed} failed ({rate:.1f}/s)")
            continue

        rel_path = str(en_file.relative_to(project_root))
        print(f"  [{i+1}/{total}] {rel_path} ...", end=" ", flush=True)

        if translate_file(en_file):
            done += 1
            elapsed = time.time() - start_time
            rate = (done + skipped) / elapsed if elapsed > 0 else 0
            print(f"OK ({rate:.1f}/s)")
        else:
            failed += 1
            failed_files.append(rel_path)
            print("FAILED")

        # Small delay between requests
        time.sleep(0.3)

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"Done! {done} translated, {skipped} skipped, {failed} failed")
    print(f"Total time: {elapsed/60:.1f} min ({elapsed:.0f}s)")

    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for f in failed_files[:20]:
            print(f"  - {f}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files) - 20} more")


if __name__ == "__main__":
    main()
