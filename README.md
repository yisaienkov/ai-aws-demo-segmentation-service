# AI AWS Demo Segmentation Service

## Build and Run

```bash

$ docker build -t demo_segmentation_service -f Dockerfile .

$ docker run -p 8000:8000 -e BUCKET=XXX -e ACCESS_KEY=XXX -e SECRET_KEY=XXX demo_segmentation_service
```