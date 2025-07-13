#!/bin/bash

# Tạo thư mục dữ liệu
mkdir -p coco_data
cd coco_data

# Tạo cấu trúc thư mục
mkdir -p train2017
mkdir -p val2017
mkdir -p test2017
mkdir -p annotations

echo "Downloading COCO 2017 dataset..."

# Download training images
echo "Downloading training images..."
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip -q train2017.zip
rm train2017.zip

# Download validation images
echo "Downloading validation images..."
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip
rm val2017.zip

# Download test images
echo "Downloading test images..."
wget -c http://images.cocodataset.org/zips/test2017.zip
unzip -q test2017.zip
rm test2017.zip

# Download annotations
echo "Downloading annotations..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip
rm annotations_trainval2017.zip

# Download test annotations
wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip
unzip -q image_info_test2017.zip
rm image_info_test2017.zip

echo "Dataset download completed!"
echo "Directory structure:"
ls -la

cd ..
echo "COCO dataset is ready in coco_data/ directory"