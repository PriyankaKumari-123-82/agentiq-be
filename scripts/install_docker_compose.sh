#!/bin/bash
# Install Docker if not already installed
if ! command -v docker &> /dev/null
then
    echo "Docker not found, installing..."
    sudo yum update -y
    sudo yum install -y docker
    sudo service docker start
    sudo usermod -a -G docker ec2-user # Add ec2-user to docker group
    echo "Docker installed. Please ensure the instance is restarted or log out/in for group changes to take effect."
fi

# Install Docker Compose if not already installed
if ! command -v docker-compose &> /dev/null
then
    echo "Docker Compose not found, installing..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "Docker Compose installed."
fi

# Ensure docker is running
sudo service docker start