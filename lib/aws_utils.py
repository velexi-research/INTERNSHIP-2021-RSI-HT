import boto3
import os

PUBLIC_KEY = ''
SECRET_KEY = ''
BUCKET = ''


def get_bucket_objects():
    s3 = boto3.Session(
        aws_access_key_id=PUBLIC_KEY,
        aws_secret_access_key=SECRET_KEY,
    ).client('s3')

    response = s3.list_objects(Bucket=BUCKET)
    locations = []
    for entry in response['Contents']:
        entry_key = entry['Key']
        if entry_key[-3:] == '.pt':
            locations.append(entry_key)

    return locations


def download_objects(locations):
    s3 = boto3.Session(
        aws_access_key_id=PUBLIC_KEY,
        aws_secret_access_key=SECRET_KEY,
    ).resource('s3')

    for entry_key in locations:
        if not os.path.exists('../' + entry_key):
            s3.meta.client.download_file(BUCKET, entry_key, '../' + entry_key)


def main():
    locations = get_bucket_objects()
    download_objects(locations)


if __name__ == '__main__':
    main()
