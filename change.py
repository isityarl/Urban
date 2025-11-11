import re

with open("getObservation.txt", "r") as f:
    text = f.read()

text = re.sub(r"\broot\.", "", text)
pattern = r"(?:root\.)?(road\d+)\.nCars\((true|false)\)"
replacement = r"getDensity(\1, \2)"
new_text = re.sub(pattern, replacement, text)

with open("getObservation.txt", "w") as f:
    f.write(new_text)

print('Done')
