rm -rf __pycache__/
rm .DS_Store
find . -type d -name  "__pycache__" -exec rm -r {} +
rm -r build dist *.egg-info