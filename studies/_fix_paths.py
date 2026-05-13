"""Fix SRC_ROOT references in studies/ directory."""
import os

root = r"c:\Users\Hellx\Documents\Programming\python\Project\iron\bc\GibbsQ"
studies_dir = os.path.join(root, "studies")
count = 0

old_block = (
    'SRC_ROOT = PROJECT_ROOT / "src"\n'
    "if str(PROJECT_ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(PROJECT_ROOT))\n"
    "if str(SRC_ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(SRC_ROOT))\n"
)

new_block = (
    "if str(PROJECT_ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(PROJECT_ROOT))\n"
)

for dirpath, dirnames, filenames in os.walk(studies_dir):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for fn in filenames:
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        fpath = os.path.join(dirpath, fn)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        original = content
        content = content.replace(old_block, new_block)
        if content != original:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            count += 1
            print(f"Updated: {os.path.relpath(fpath, root)}")

print(f"Total: {count}")
