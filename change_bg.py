# #sk-149a52ca5f9e4067a4620f8af8ba02b5
import requests
import json   
import time
import os
img_url='https://onko.oss-cn-nanjing.aliyuncs.com/%E9%98%B3%E6%80%A7__1_-removebg.png'
# # response = requests.get(img_url)
# # with open("result.jpg","wb") as f:
# #     f.write(response.content)

Header={
    "Content-Type": "application/json",
    "Authorization":"Bearer sk-149a52ca5f9e4067a4620f8af8ba02b5",
    "X-DashScope-Async":"enable"
}

Data={
    "model":"wanx-background-generation-v2",
    "input":{
        "base_image_url":img_url,
        "ref_prompt":"random scene in daily life"
    },
    "parameters":{
        "n":"4",
        "ref_prompt_weight":0.8,
        "scene_type":"ROOM"
    }
}

# api="https://dashscope.aliyuncs.com/api/v1/services/aigc/background-generation/generation/"

# response = requests.post(api, headers=Header, data=json.dumps(Data))

# task_id=response.json()['output']['task_id']
# print(f"task_id:{task_id}")

# #GET https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}

cnt=0
while True:
    print(f"{cnt} times checking, waiting for task to complete...")
    cnt+=1
    response = requests.get(f"https://dashscope.aliyuncs.com/api/v1/tasks/4bc2385b-4b64-41b9-a52b-f6fb5eef804b", headers=Header)
    print(json.dumps(response.json(), indent=4))
    if response.json()['output']['task_status'] == 'SUCCEEDED':
        break
    time.sleep(2)
    
bg_changed_dir="changed_bg"
if not os.path.exists(bg_changed_dir):
    os.mkdir(bg_changed_dir)
    
imgs=response.json()['output']['results']
for img in imgs:
    img_url=img['url']
    print(f"Downloading {img_url}")
    response = requests.get(img_url)
    filename = os.path.basename(img_url.split('?')[0])
    with open(os.path.join(bg_changed_dir, filename), 'wb') as f:
        f.write(response.content)
    


