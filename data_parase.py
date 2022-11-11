import json
class_ids=[]
with open('/home/luoyang02/model-zoo/longtail-det/ACSL-main/data/lvis/lvis_v0.5_train.json','r',encoding='utf8')as fp:
    data=json.load((fp))
    print('ok')
    for i in range(len(data['annotations'])):
        annotation=annotations[i]
        class_id=annotation['category_id']
        class_ids.append((class_id))
