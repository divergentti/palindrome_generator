# Builder for Palindrome Generator

## Building builder image with Docker
This will create builder image for palindrome generator with Docker:
```
sudo docker build -t palindrome-generator:0.1 .
```

## Running the builder
This will build the palindrome generator into `/opt/palindrome_generator` folder:
```
sudo docker run --name palindrome_generator -v /opt/palindrome_generator:/app/dist palindrome-generator:0.1
```
_NOTE:_ Building may consume excessive amount of resources.

## Running Palindrome Generator
```
# Change permissions for PalindromiPeli.bin
sudo chown $USER PalindromiPeli.bin

# Move PalindromiPeli.bin into folder where /data directory is
sudo mv PalindromiPeli.bin {path/to/palindrome-generator/}

# Run PalindromiPeli.bin
./PalindromiPeli.bin
```

## Cleaning Docker
```
# List docker containers
sudo docker ps -a
sudo docker rm palindrome_generator
# List docker images
sudo docker images
sudo docker rmi palindrome_generator:0.1
```
