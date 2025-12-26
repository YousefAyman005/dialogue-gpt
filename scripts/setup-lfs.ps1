#!/usr/bin/env pwsh
# Setup Git LFS and track common model/tokenizer file patterns
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.bin"
git lfs track "movie_tokenizer.model"
git lfs track "movie_tokenizer.vocab"
git add .gitattributes
Write-Host "Git LFS initialized and patterns tracked. Commit .gitattributes and push."
