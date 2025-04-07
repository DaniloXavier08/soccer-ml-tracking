import requests

print("Downloading COCO class labels...")
url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
response = requests.get(url)

print("Response status code:", response.status_code)
if response.status_code != 200:
    print("Failed to download COCO class labels")
    exit()

with open("weights/coco.names", "w") as file:
    file.write(response.text)
    print("COCO class labels downloaded successfully.")