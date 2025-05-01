```shell
 docker buildx build --platform linux/amd64,linux/arm64 -t nazarmedykh/robocasting:v2 --push .
```


```shell
docker run -d --name robocasting -p 8501:8501   -v robocasting_data:/app/data  --restart unless-stopped nazarmedykh/robocasting:v2
```