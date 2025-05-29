""" Script to auto-update README with a project tree """

import os
import sys
from pathlib import Path

README_PATH = Path("README.md")
TREE_START = "<!-- TREE START -->"
TREE_END = "<!-- TREE END -->"
EXCLUDE_DIRS = {"venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}

def generate_tree(root=".", prefix=""):
    tree = []
    try:
        entries = sorted(os.listdir(root))
        entries = [
            e for e in entries
            if e not in EXCLUDE_DIRS
            and (not e.startswith(".") or e == ".github")
        ]

        files = [e for e in entries if os.path.isfile(os.path.join(root, e))]
        dirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]

        for file in files:
            tree.append(f"{prefix}├── {file}")
        for i, dir in enumerate(dirs):
            path = os.path.join(root, dir)
            branch = "└──" if i == len(dirs) - 1 else "├──"
            tree.append(f"{prefix}{branch} {dir}/")
            subtree = generate_tree(path, prefix + ("    " if i == len(dirs) - 1 else "│   "))
            tree.extend(subtree)

    except Exception as e:
        print(f"❌ Error generating tree: {e}")
        return []

    return tree

def inject_tree_into_readme(tree_lines):
    if not README_PATH.exists():
        print("❌ README.md not found in project root.")
        return 1

    try:
        with README_PATH.open("r", encoding="utf-8") as f:
            content = f.read()

        new_tree = (
            f"{TREE_START}\n"
            "📁 Project Structure\n\n"
            "solar-challenge-week1/\n"
            + "\n".join(tree_lines)
            + f"\n{TREE_END}"
        )

        if TREE_START in content and TREE_END in content:
            pre = content.split(TREE_START)[0]
            post = content.split(TREE_END)[-1]
            new_content = pre + new_tree + post
        else:
            new_content = content.strip() + "\n\n" + new_tree + "\n"

        with README_PATH.open("w", encoding="utf-8") as f:
            f.write(new_content)

        print("✅ README.md successfully updated with the latest tree.")
        return 0

    except Exception as e:
        print(f"❌ Failed to update README.md: {e}")
        return 1

if __name__ == "__main__":
    print("🔄 Generating project tree and updating README...")
    tree = generate_tree()
    if tree:
        sys.exit(inject_tree_into_readme(tree))
    else:
        print("❌ No tree structure generated. Nothing to update.")
        sys.exit(1)
# This script generates a tree structure of the project directory and injects it into the README.md file.
