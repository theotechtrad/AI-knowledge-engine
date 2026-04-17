import codecs

# Read with utf-8-sig first, fallback to utf-16
try:
    content = open('requirements.txt', 'r', encoding='utf-8-sig').read()
except UnicodeDecodeError:
    content = open('requirements.txt', 'r', encoding='utf-16').read()

lines = content.splitlines()
cleaned = []

for line in lines:
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    if line.startswith('torch=='):
        continue
    if line.startswith('-e git+'):
        line = line[3:]
    cleaned.append(line)

# Write as clean UTF-8 with Unix line endings
with open('requirements_clean.txt', 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(cleaned))

print('Done cleaning requirements')