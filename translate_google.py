import json, sys, time
from googletrans import Translator

data = json.load(open(sys.argv[1], encoding="utf-8"))
langs = {1: "de", 2: "fr", 3: "cs"}
tr = Translator()

for offset, lang in langs.items():
    print(f"翻译 → {lang}...")
    for g in range(len(data) // 4):
        try:
            data[g * 4 + offset] = tr.translate(data[g * 4], dest=lang).text
        except Exception:
            time.sleep(2)
            try:
                data[g * 4 + offset] = tr.translate(data[g * 4], dest=lang).text
            except Exception:
                pass  # 保留原文
        if g % 100 == 0 and g > 0:
            print(f"  {g}/{len(data)//4}")
            time.sleep(0.5)

json.dump(data, open(sys.argv[1].replace(".json", "_translated.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("完成！")
