#!/bin/bash

# Setup script for llama.cpp and TinyLlama model
# Run this script to prepare the environment for facts dataset expansion

echo "=== Setting up llama.cpp for Facts Dataset Expansion ==="

# Check if we're in the right directory
if [ ! -f "expand_facts_dataset.py" ]; then
    echo "ERROR: Please run this script from the memory-ai directory"
    exit 1
fi

# Create models directory
mkdir -p models
cd models

echo "1. Installing llama.cpp..."

# Clone and build llama.cpp
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp

# Use CMake build system (newer versions of llama.cpp)
mkdir -p build
cd build
cmake .. -DLLAMA_CURL=OFF
make llama-server

# Check if build was successful
if [ ! -f "llama-server" ]; then
    echo "ERROR: Failed to build llama-server"
    exit 1
fi

echo "2. Downloading TinyLlama model..."

# Download TinyLlama model
if [ ! -f "../tinyllama-1.1b-chat-v1.0.Q4_0.gguf" ]; then
    wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
         -O ../tinyllama-1.1b-chat-v1.0.Q4_0.gguf
fi

cd ../..

echo "3. Testing llama.cpp server..."

# Test the server
./models/llama.cpp/llama-server \
    --model models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    --ctx-size 2048 \
    --threads 4 \
    --port 8080 \
    --health &
SERVER_PID=$!

sleep 5

# Test health endpoint
if curl -s http://localhost:8080/health > /dev/null; then
    echo "✅ llama.cpp server is running successfully!"
    kill $SERVER_PID
else
    echo "❌ Failed to start llama.cpp server"
    kill $SERVER_PID
    exit 1
fi

echo ""
echo "=== Setup Complete! ==="
echo "You can now run the facts expansion script:"
echo "python expand_facts_dataset.py"
echo ""
echo "Note: The server will start automatically when you run the expansion script."