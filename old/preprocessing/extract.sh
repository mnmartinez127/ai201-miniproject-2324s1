wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/gh56wbsnj5-1.zip -P preprocessing/datasets
unzip preprocessing/datasets/gh56wbsnj5-1.zip -d preprocessing/datasets/
unzip preprocessing/datasets/Coconut\ Tree\ Disease\ Dataset/Coconut\ Tree\ Disease\ Dataset.zip -d preprocessing/datasets/
rm -rf preprocessing/datasets/Coconut\ Tree\ Disease\ Dataset/Coconut\ Tree\ Disease\ Dataset.zip && rm -rf preprocessing/datasets/gh56wbsnj5-1.zip
