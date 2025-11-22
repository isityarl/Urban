import os

DIR = "/home/yarl/Desktop/git/Urban/baseline/"

files_and_tags = {
    "edgeData.xml": "</meandata>",
    "stats.xml": "</statistics>",
    "stopinfos.xml": "</stops>",
    "tripinfos.xml": "</tripinfos>", 
}

def fix_xml_file(filepath, closing_tag):
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        data_str = data.decode("utf-8", errors="ignore")
        if data_str.strip().endswith(closing_tag):
            print(f"{filepath} is complete.")
            return
        with open(filepath, "a", encoding="utf-8") as f:
            f.write("\n" + closing_tag + "\n")
        print(f"Fixed {filepath}.")
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")

for filename, closing_tag in files_and_tags.items():
    filepath = os.path.join(DIR, filename)
    fix_xml_file(filepath, closing_tag)
