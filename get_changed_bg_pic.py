import requests
import os
import json

def get_changed_bg_pic():
    """
    获取更改了背景的图片并保存到本地
    :param img_path: 图片的路径
    :param url: 图片的url
    :return: None
    """
    # 下载图片
    url_1='https://img.logosc.cn/api/paint'
    url_2='https://img.logosc.cn/api/paint?taskId='
    # 设置cookie
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "Connection": "keep-alive",
        "Content-Length":"130",
        "Content-Type": "application/json",
        "Cookie": "auth=1; token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOlwvXC9hcGkuaW1nLmxvZ29zYy5jblwvdjFcL2dhdGV3YXlcL3dlY2hhdFwvcXJjb2RlXC9jaGVja1wvY2xpZW50JTdDZjYzOWY3NzktM2FjMS00MjhlLTliODItYzVmZjBlOTFiNTFkIiwiaWF0IjoxNzI0MTQxNjM4LCJleHAiOjE3MjY3MzM2MzgsIm5iZiI6MTcyNDE0MTYzOCwianRpIjoiNHV3OGNSeFBlTDdDQW1UWiIsInN1YiI6NDUwMTE5LCJwcnYiOiIyM2JkNWM4OTQ5ZjYwMGFkYjM5ZTcwMWM0MDA4NzJkYjdhNTk3NmY3Iiwicm9sZSI6ImFwaSJ9.Vqgx9Ahoxx99ktmArH64g9dc6CccBqBr9H4lOzXgy2g",
        "Host": "img.logosc.cn",
        "Referer": "https://img.logosc.cn/ai-edit",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"99\", \"Microsoft Edge\";v=\"127\", \"Chromium\";v=\"127\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0"
        }

    
    payload={
        "imgUrl":"https://gaitu.oss-cn-hangzhou.aliyuncs.com/tmp/3f36a4b88dcd68c35638b114f985c412.png",
        "text":"生活中的随机场景"
    }
        
    response = requests.post(url_1, headers=headers,params=payload)
    print(response.text)
    requestsId=response.json()['requestId']
    url_2=url_2+requestsId
    print(f'url_2:{url_2}')
    response=requests.get(url_2,headers=headers)
    img_url = response.json()['data']['resultList'][0]['result']
    try:
        response = requests.get(img_url, timeout=10)  # 设置超时时间为10秒
        response.raise_for_status()  # 检查是否成功

        with open("downloaded_image.jpg", "wb") as file:
            file.write(response.content)
        print("图片下载成功并保存为 downloaded_image.jpg")
    except requests.exceptions.RequestException as e:
        print(f"图片下载失败: {e}")
    

get_changed_bg_pic()