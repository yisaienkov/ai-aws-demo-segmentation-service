# AI AWS Demo Segmentation Service

## Build and Run

```bash

$ docker build -t demo_segmentation_service -f Dockerfile .

$ docker run -p 8000:8000 -e BUCKET=XXX -e AWS_ACCESS_KEY_ID=XXX -e AWS_SECRET_ACCESS_KEY=XXX demo_segmentation_service
```
