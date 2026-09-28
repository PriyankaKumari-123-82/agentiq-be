#!/bin/bash
# Stop the running Docker Compose services
echo "Stopping existing Docker Compose services..."
if [ -f /opt/codedeploy-agent/deployment-root/docker-compose.yml ]; then
    cd /opt/codedeploy-agent/deployment-root/
    docker-compose down || true # '|| true' prevents script from failing if services aren't running
else
    echo "No docker-compose.yml found to stop."
fi