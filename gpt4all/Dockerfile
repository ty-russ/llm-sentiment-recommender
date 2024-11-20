# Use the Ubuntu image with multi-platform support
FROM --platform=linux/amd64 ubuntu

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required packages
RUN apt -y update && apt -y upgrade && apt -y install wget

# Download the GPT4All installer
RUN wget https://gpt4all.io/installers/gpt4all-installer-linux.run && \
    chmod +x gpt4all-installer-linux.run

# Install necessary dependencies
RUN apt install -y libfontconfig1 libdbus-1-3 libxcb-glx0 libx11-xcb1 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
    libxcb-sync1 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 \
    libgl1 xdg-user-dirs

# Install Qt dependencies individually (since qt5-default is no longer packaged)
RUN apt install -y qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools

# Create a non-root user and switch to it for security
RUN useradd -m user && \
    chown -R user:user /home/user
USER user
WORKDIR /home/user

# Ensure user directories are set up for GUI apps
RUN xdg-user-dirs-update

# Set the command to check if GPT4All is installed and run it
# This will install GPT4All if it's not already installed
CMD test -f /home/user/gpt4all/bin/chat || ./gpt4all-installer-linux.run && /home/user/gpt4all/bin/chat
