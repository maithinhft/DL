import kagglehub

# Download latest version

def download_dataset():

    path = kagglehub.dataset_download("awsaf49/coco-2017-dataset")

    print("Path to dataset files:", path)
    return path