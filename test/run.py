import scales_nlp
import os


ucid = "azd;;3:18-cv-08134"

os.system(f"scales-nlp download '{ucid}'")
os.system("scales-nlp parse")
os.system("scales-nlp predict")

docket = scales_nlp.Docket.from_ucid(ucid)

print(docket.ucid)
print(docket.court)
print(docket.header.keys())
for entry in docket:
    print()
    print(entry.row_number, entry.entry_number, entry.date_filed)
    print(entry.text)
    print(entry.event)
    print(entry.labels)
