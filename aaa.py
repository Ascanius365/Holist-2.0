import re

file_path = "consolidation/run_reasoning.py"

with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# 2. Schritt: Ersetzt alle verbleibenden auskommentierten '# print(' durch aktives 'logger.info('
code = re.sub(r'#print\(', 'logger.info(', code)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ Alle Prints erfolgreich in logger.info konvertiert!")