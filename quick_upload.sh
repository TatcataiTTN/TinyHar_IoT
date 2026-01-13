#!/bin/bash

# Quick GitHub Upload Script
# Run this after creating repository on GitHub

echo "🚀 TinyHAR - Quick GitHub Upload"
echo "================================"
echo ""

# Check if repository exists on GitHub
read -p "Have you created the repository on GitHub? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Please create repository first:"
    echo "1. Go to: https://github.com/new"
    echo "2. Repository name: TinyHAR"
    echo "3. Description: Human Activity Recognition on ESP32 with TensorFlow Lite Micro"
    echo "4. Choose Public or Private"
    echo "5. DO NOT add README, .gitignore, or license"
    echo ""
    echo "After creating, run this script again."
    exit 0
fi

echo ""
read -p "Enter your GitHub username: " github_username

if [ -z "$github_username" ]; then
    echo "❌ Username cannot be empty"
    exit 1
fi

echo ""
echo "📦 Repository URL: https://github.com/$github_username/TinyHAR"
echo ""

# Add remote
echo "🔗 Adding remote origin..."
git remote remove origin 2>/dev/null
git remote add origin "https://github.com/$github_username/TinyHAR.git"

if [ $? -eq 0 ]; then
    echo "✅ Remote added successfully"
else
    echo "❌ Failed to add remote"
    exit 1
fi

echo ""
echo "🚀 Pushing to GitHub..."
echo ""

# Push to GitHub
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✅ SUCCESS! Repository uploaded to GitHub!"
    echo "================================================"
    echo ""
    echo "🎉 Your repository is now live at:"
    echo "   https://github.com/$github_username/TinyHAR"
    echo ""
    echo "📊 Repository contains:"
    echo "   - 18 files"
    echo "   - 4,930+ lines of documentation"
    echo "   - 70+ code samples"
    echo "   - 50+ references"
    echo ""
    echo "🎨 Next steps:"
    echo "   1. Visit your repository"
    echo "   2. Add topics: machine-learning, esp32, tensorflow-lite, iot"
    echo "   3. Star your own repo ⭐"
    echo "   4. Share with others!"
    echo ""
else
    echo ""
    echo "❌ Push failed. This might be due to:"
    echo "   1. Authentication required"
    echo "   2. Repository doesn't exist"
    echo "   3. Network issues"
    echo ""
    echo "💡 Try these solutions:"
    echo ""
    echo "Option 1: Use Personal Access Token"
    echo "   1. Go to: https://github.com/settings/tokens"
    echo "   2. Generate new token (classic)"
    echo "   3. Select 'repo' scope"
    echo "   4. Copy token"
    echo "   5. Run: git push -u origin main"
    echo "   6. Use token as password"
    echo ""
    echo "Option 2: Use GitHub CLI"
    echo "   brew install gh"
    echo "   gh auth login"
    echo "   git push -u origin main"
    echo ""
fi

