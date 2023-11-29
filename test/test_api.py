import requests

versions = [
    "2",
    "3"
]


for v in versions:

    # URL endopint
    json_response = requests.post(
        f'http://localhost:8000/tf/seal_detector/{v}/url',
        headers={
            "content-type": "application/json"
        },
        json={
            "url": "https://s3.us-west-2.amazonaws.com/webcoos/media/sources/webcoos/groups/noaa/assets/tmmc_prls/feeds/raw-video-data/products/one-minute-stills/elements/2023/02/20/tmmc_prls-2023-02-20-015457Z.jpg",
        }
    )

    # Image upload endpoint
    with open('inputs/seal-01.jpg', 'rb') as f:
        json_response = requests.post(
            f'http://localhost:8000/tf/seal_detector/{v}/upload',
            files={
                'file': ('seal-01.jpg', f)
            }
        )
