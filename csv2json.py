# converting the CSV of capitals to a JSON file
# thanks to: https://www.geeksforgeeks.org/python/convert-csv-to-json-using-python/

import csv
import jsonlines

with open('national_capitals_no_duplicates.csv', mode='r', newline='', encoding='utf-8') as csvfile, jsonlines.open('national_capitals.jsonl', mode='w') as writer:
    for row in csv.DictReader(csvfile):
        writer.write(row)