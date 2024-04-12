#!/bin/bash

# Function to check if the OS is Windows
is_windows() {
    if [[ "$(uname -s)" =~ "MINGW" || "$(uname -s)" =~ "MSYS" || "$(uname -o)" =~ "Msys" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to download the zip file
download_zip() {
    local URL="$1"
    local ZIP_FILE=$(basename "$URL")
    if is_windows; then
        curl -o "$ZIP_FILE" "$URL" || { echo "Error: Failed to download $ZIP_FILE"; exit 1; }
    else
        wget "$URL" || { echo "Error: Failed to download $ZIP_FILE"; exit 1; }
    fi
}

# Function to unzip the file
unzip_file() {
    local ZIP_FILE="$1"
    if is_windows; then
        powershell Expand-Archive -Path "$ZIP_FILE" -DestinationPath . || { echo "Error: Failed to unzip $ZIP_FILE"; exit 1; }
    else
        unzip "$ZIP_FILE" || { echo "Error: Failed to unzip $ZIP_FILE"; exit 1; }
    fi
}

# Function to delete the zip file
delete_zip() {
    local ZIP_FILE="$1"
    if is_windows; then
        del "$ZIP_FILE" || { echo "Error: Failed to delete $ZIP_FILE"; exit 1; }
    else
        rm "$ZIP_FILE" || { echo "Error: Failed to delete $ZIP_FILE"; exit 1; }
    fi
}

# URL of the zip file
URL="https://zenodo.org/records/10964223/files/data.zip?download=1"

# File name of the zip file
ZIP_FILE=$(basename "$URL")

# Download the zip file
download_zip "$URL"

# Unzip the file
unzip_file "$ZIP_FILE"

# Delete the zip file
delete_zip "$ZIP_FILE"

# Optional: Print a message indicating completion
echo "Downloaded the $ZIP_FILE folder successfully"
