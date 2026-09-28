#!/bin/bash
set -eux # Exit immediately if a command exits with a non-zero status, print commands

TARGET_BASE_DIR="/home/ubuntu/ai_agent/pb-ai-agent"
TARGET_SCRIPTS_DIR="$TARGET_BASE_DIR/scripts"

# CodeDeploy provides these env vars
DEPLOYMENT_ARCHIVE_ROOT="/opt/codedeploy-agent/deployment-root/${DEPLOYMENT_GROUP_ID}/${DEPLOYMENT_ID}/deployment-archive"

SOURCE_SCRIPT_PATH="$DEPLOYMENT_ARCHIVE_ROOT/scripts/start_docker_compose.sh"

echo "--- CodeDeploy BeforeInstall Hook: Custom Script Sync ---"
echo "Deployment Archive Root: $DEPLOYMENT_ARCHIVE_ROOT"
echo "Source start_docker_compose.sh path in archive: $SOURCE_SCRIPT_PATH"
echo "Target scripts directory on instance: $TARGET_SCRIPTS_DIR"

# Ensure the target scripts directory exists
mkdir -p "$TARGET_SCRIPTS_DIR"
echo "Verified/created target scripts directory: $TARGET_SCRIPTS_DIR"

if [ -f "$SOURCE_SCRIPT_PATH" ]; then
    echo "Attempting to rsync $SOURCE_SCRIPT_PATH to $TARGET_SCRIPTS_DIR"
    rsync -up --chmod=u+rwx "$SOURCE_SCRIPT_PATH" "$TARGET_SCRIPTS_DIR/"
    echo "rsync command executed."
    # Ensure the copied script is executable
    chmod +x "$TARGET_SCRIPTS_DIR/start_docker_compose.sh"
    echo "Permissions set for $TARGET_SCRIPTS_DIR/start_docker_compose.sh"
else
    echo "Warning: $SOURCE_SCRIPT_PATH not found in deployment archive. Skipping rsync for this file."
fi

echo "--- Custom Script Sync Completed ---"