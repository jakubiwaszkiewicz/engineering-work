import json
file_path = 'data.txt'
json_data = []
list_series_of_labels = []

with open(file_path, 'r') as file:
    series_of_labels = file.read().splitlines()

for labels in series_of_labels:
    label = list(labels)
    list_series_of_labels.append(label)

i=1

for bee_frame in list_series_of_labels:
  json_data.append({
    'id': i,
    'have_honey': bee_frame[0],
    'have_seal': bee_frame[1],
    'hardness': str(int(bee_frame[2])-1),
  })
  i += 1

output_json = json.dumps(json_data, indent=4)

with open("labels.json", "w") as outfile:
    outfile.write(output_json)