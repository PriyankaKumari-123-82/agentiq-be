#!/bin/bash
echo "Verifying application status..."

# Define the health endpoint and expected status code
HEALTH_ENDPOINT="http://localhost:8000/health"
EXPECTED_STATUS=200
MAX_RETRIES=10
RETRY_INTERVAL=5 # seconds

# Loop to check health with retries
for i in $(seq 1 $MAX_RETRIES); do
    echo "Attempt $i/$MAX_RETRIES: Checking application health at $HEALTH_ENDPOINT..."
    
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_ENDPOINT)

    if [ "$HTTP_STATUS" -eq "$EXPECTED_STATUS" ]; then
        echo "Application is healthy (HTTP Status: $HTTP_STATUS)."
        exit 0 # Success
    else
        echo "Application not yet healthy (HTTP Status: $HTTP_STATUS). Retrying in $RETRY_INTERVAL seconds..."
        sleep $RETRY_INTERVAL
    fi
done

echo "Error: Application did not become healthy after $MAX_RETRIES attempts."
exit 1 # Failure - this will trigger a CodeDeploy rollback if configured