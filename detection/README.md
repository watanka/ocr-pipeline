## STD: 문서 이미지에서 텍스트 영역을 탐지함

input: 이미지 base64 인코딩 문자열
```
{
    "file_name": "test_image.jpg",
    "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
}
```


output
```
[
    {
        "id": "test_image_1",
        "file_name": {filename},
        "image": "random_crop_image1"
    }, ...
]

```


- 모델, 전처리, 후처리 로직은 생략.