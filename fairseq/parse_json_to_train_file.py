import json
fr = open('total.json','r')
fw = open('len_label.txt','w')
json_content = json.load(fr)
print(len(json_content))

for item in json_content:
    fw.write(' '.join(map(str,list(map(int,json_content[item]['word_trans']))))+'\n')
fr.close()
fw.close()