#!/usr/bin/env python3
"""Manage MY_PROGRESS.md for the AI Engineering from Scratch curriculum."""

import re
import sys
from datetime import date
import json
from pathlib import Path
from typing import Optional



def _sync_progress_json(progress_path: Path):
    """Sync completed lessons to site/progress.json for the localhost website."""
    content = progress_path.read_text(encoding="utf-8")
    completed_entries = re.findall(r"^- ✅ Phase (\d+) / ([\w-]+)", content, re.MULTILINE)
    
    # Find current phase from progress file
    phase_match = re.search(r"\*\*当前 Phase：\*\* (\d+)", content)
    lesson_match = re.search(r"\*\*当前课程：\*\* ([\w-]+)", content)
    
    json_path = progress_path.parent / "site" / "progress.json"
    if not json_path.parent.exists():
        return
    
    data = {}
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            pass
    
    # Build completed list from MY_PROGRESS.md
    completed = []
    for phase_num, lesson_slug in completed_entries:
        # Find the actual phase directory
        phase_dirs = sorted(progress_path.parent.glob(f"phases/{phase_num.zfill(2)}-*"))
        if phase_dirs:
            phase_dir_name = phase_dirs[0].name
            completed.append(f"phases/{phase_dir_name}/{lesson_slug}")
    
    data["completed"] = completed
    data["updatedAt"] = date.today().isoformat()
    
    if phase_match:
        data["currentPhase"] = int(phase_match.group(1))
    if lesson_match:
        data["currentLesson"] = lesson_match.group(1)
    
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"Synced progress.json: {len(completed)} completed")


def find_progress_file(start_dir: str = ".") -> Optional[Path]:
    """Find MY_PROGRESS.md in current or parent directories."""
    d = Path(start_dir).resolve()
    for _ in range(5):
        f = d / "MY_PROGRESS.md"
        if f.exists():
            return f
        if d.parent == d:
            break
        d = d.parent
    return None


def mark_complete(progress_path: Path, phase: str, lesson: str, lesson_name: str = "") -> bool:
    """Mark a lesson as completed."""
    content = progress_path.read_text(encoding="utf-8")
    today = date.today().isoformat()

    entry = f"- ✅ Phase {phase} / {lesson}"
    if lesson_name:
        entry += f" — {lesson_name}"
    entry += f" ({today})"

    # Sync progress.json regardless of existing records
    _sync_progress_json(progress_path)

    if entry in content:
        print(f"Already recorded: {entry}")
        return False

    # Insert after "## 已完成" line
    lines = content.split("\n")
    new_lines = []
    inserted = False
    for line in lines:
        new_lines.append(line)
        if not inserted and line.strip() == "## 已完成":
            new_lines.append(entry)
            inserted = True

    if not inserted:
        new_lines.append("\n## 已完成\n")
        new_lines.append(entry)

    progress_path.write_text("\n".join(new_lines), encoding="utf-8")

    # Sync to site/progress.json for localhost website
    _sync_progress_json(progress_path)

    print(f"Marked complete: {entry}")

    # Update current course
    match = re.match(r"(\d+)-", lesson)
    if match:
        lesson_num = int(match.group(1))
        next_lesson = f"{lesson_num + 1:02d}"
        content2 = progress_path.read_text(encoding="utf-8")
        content2 = re.sub(
            r"\*\*当前课程：\*\* .+",
            f"**当前课程：** {next_lesson}-（待查课程名）",
            content2
        )
        progress_path.write_text(content2, encoding="utf-8")

    return True


def show_summary(progress_path: Path):
    """Print a summary of progress."""
    content = progress_path.read_text(encoding="utf-8")
    completed = re.findall(r"^- ✅ (.+)$", content, re.MULTILINE)
    phase_match = re.search(r"\*\*当前 Phase：\*\* (.+)", content)
    lesson_match = re.search(r"\*\*当前课程：\*\* (.+)", content)

    print("=" * 50)
    print("学习进度摘要")
    print("=" * 50)
    if phase_match:
        print(f"当前阶段: {phase_match.group(1)}")
    if lesson_match:
        print(f"当前课程: {lesson_match.group(1)}")
    print(f"已完成: {len(completed)} 课")
    for c in completed:
        print(f"  {c}")
    print("=" * 50)


def main():
    progress = find_progress_file()
    if not progress:
        print("Error: MY_PROGRESS.md not found. Are you in the project directory?")
        sys.exit(1)

    if len(sys.argv) < 2:
        show_summary(progress)
        return

    cmd = sys.argv[1]

    if cmd == "complete" and len(sys.argv) >= 4:
        phase = sys.argv[2]
        lesson = sys.argv[3]
        name = sys.argv[4] if len(sys.argv) > 4 else ""
        mark_complete(progress, phase, lesson, name)
    elif cmd == "summary":
        show_summary(progress)
    elif cmd == "path":
        print(progress)
    else:
        print("Usage:")
        print("  python update_progress.py                     # show summary")
        print("  python update_progress.py complete <phase> <lesson> [name]")
        print("  python update_progress.py path")


if __name__ == "__main__":
    main()
