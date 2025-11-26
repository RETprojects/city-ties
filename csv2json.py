# converting the CSV of capitals to a JSON file
# thanks to: https://www.geeksforgeeks.org/python/convert-csv-to-json-using-python/

import csv
import json

with open('national_capitals_no_duplicates.csv', mode='r', newline='', encoding='utf-8') as csvfile:
    data = list(csv.DictReader(csvfile))

with open('national_capitals.json', mode='w', encoding='utf-8') as jsonfile:
    json.dump(data, jsonfile, indent=4)