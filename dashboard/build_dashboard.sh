#!/bin/bash

echo "🚀 Building Catalyst Vector Alpha Dashboard..."

# Navigate to dashboard directory
cd "$(dirname "$0")" || exit 1

echo "📦 Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "🔨 Building React application..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Failed to build React application"
    exit 1
fi

echo "📁 Copying build files to static directory..."
# Create static dashboard directory if it doesn't exist
mkdir -p ../static/dashboard

# Copy build files
cp -r build/* ../static/dashboard/

echo "✅ Dashboard build completed successfully!"
echo "📍 Files copied to: static/dashboard/"
echo "🌐 Access the dashboard at: http://localhost:5000/dashboard"