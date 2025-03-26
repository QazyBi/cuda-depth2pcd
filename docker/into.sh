# get the container id by image name
CONTAINER_ID=$(docker ps -q --filter "ancestor=pointcloud_fusion_env")

# enter the container
docker exec -it $CONTAINER_ID bash
