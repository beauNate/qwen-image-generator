#!/bin/bash
# ComfyUI Launcher - Double-click to start!

cd "$(dirname "$0")"
source venv/bin/activate

echo "=========================================="
echo "  Starting ComfyUI with Qwen-Image..."
echo "=========================================="
echo ""
echo "Opening browser in 5 seconds..."
echo ""

# Start ComfyUI in background
python main.py --highvram &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Open browser
open "http://127.0.0.1:8188"

echo ""
echo "ComfyUI is running!"
echo ""
echo "To generate an image:"
echo "  1. Click 'Load' in the menu (top-left)"
echo "  2. Select: qwen_workflow.json"
echo "  3. Type your prompt in the text box"
echo "  4. Click 'Queue Prompt'"
echo ""
echo "Press Ctrl+C to stop the server when done."
echo ""

# Keep script running
wait $SERVER_PID
