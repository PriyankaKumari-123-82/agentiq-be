#!/bin/bash
echo "Starting Docker Compose services..."

# Ensure jq is installed (add this at the beginning for robustness)
if ! command -v jq &> /dev/null
then
    echo "jq not found, installing..."
    sudo apt-get update && sudo apt-get install -y jq
fi


# Login to ECR - Important for pulling images
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(curl -s http://169.254.169.254/latest/dynamic/instance-identity/document | grep region | cut -d\" -f4)
$(aws ecr get-login --no-include-email --region $AWS_REGION)

# Define the application root directory where CodeDeploy copied files
APP_DEPLOY_ROOT="/opt/codedeploy-agent/deployment-root/"
cd "$APP_DEPLOY_ROOT"

# Extract the full image URI from the imagedefinitions.json
FULL_ECR_IMAGE_URI=$(jq -r '.[0].imageUri' imagedefinitions.json)
if [ -z "$FULL_ECR_IMAGE_URI" ]; then
    echo "Error: Could not extract image URI from imagedefinitions.json"
    exit 1
fi

# Extract just the image tag (e.g., 'abcdefg') from the full URI
IMAGE_TAG=$(echo "$FULL_ECR_IMAGE_URI" | awk -F ':' '{print $NF}')
if [ -z "$IMAGE_TAG" ]; then
    echo "Error: Could not extract image tag from $FULL_ECR_IMAGE_URI"
    exit 1
fi

echo "Deploying backend image with tag: $IMAGE_TAG"

cp docker-compose.yml docker-compose.yml.bak
sed -i "s|ECR_IMAGE_PLACEHOLDER|$IMAGE_TAG|g" docker-compose.yml
echo "Updated docker-compose.yml with image tag: $IMAGE_TAG"


# Use rsync to synchronize the deployment artifacts to the target directory
TARGET_APP_DIR="/home/ubuntu/ai_agent/pb-ai-agent/"

echo "Synchronizing deployment artifacts from ${APP_DEPLOY_ROOT} to ${TARGET_APP_DIR}..."
# Ensure the target directory exists and `ubuntu` user has permissions
sudo mkdir -p "$TARGET_APP_DIR"
sudo rsync -avz --delete "$APP_DEPLOY_ROOT" "$TARGET_APP_DIR"
echo "Synchronization complete."

# Change to the final application directory where docker-compose.yml is intended to run
cd "$TARGET_APP_DIR"

# Pull the specific image for your 'app' service from ECR
docker pull "$FULL_ECR_IMAGE_URI"

# Stop and remove existing containers (if any from previous deployments)
docker-compose down || true

# Bring up new containers with the updated docker-compose.yml
docker-compose up -d

echo "Docker Compose services started."