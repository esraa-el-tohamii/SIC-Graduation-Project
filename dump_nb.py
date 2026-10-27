import json

with open('sic-project.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Total cells: {len(cells)}")
print("="*80)
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    ctype = c['cell_type']
    print(f"\n--- Cell {i} [{ctype}] ---")
    print(src)
    print("---END---")
