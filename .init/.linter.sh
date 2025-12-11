#!/bin/bash
cd /home/kavia/workspace/code-generation/ai-retrieval-assistant-1892-1901/frontend
npm run build
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
   exit 1
fi

