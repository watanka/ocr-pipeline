## STR: 이미지에서 텍스트 인식

input:
```
[
    {
        "id": "image_11_crop_1",
        "file_name": "image_11.jpg",
        "image": "sdfewrwer;lkjasdklfsdfsd" # base64 encoded
    },
    {
        "id": "image_11_crop_2",
        "file_name": "image_11.jpg",
        "image": "sdfewrwer;lksdfsdfsdsdfsd" # base64 encoded
    },
]
```

output: 
```
[
    {
        "id": "image_11_crop_1",
        "file_name": "image_11.jpg",
        "text": "hello",
        "confidence": 0.95
    }, ...
]
```

- 모델, 전처리, 후처리 로직은 생략.