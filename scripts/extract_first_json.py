import sys
s = sys.stdin.read()
i = s.find('{')
if i < 0:
    sys.exit(2)
depth = 0
end = None
for k, ch in enumerate(s[i:], i):
    if ch == '{':
        depth += 1
    elif ch == '}':
        depth -= 1
    if depth == 0:
        end = k + 1
        break
if end is None:
    sys.exit(3)
print(s[i:end])
