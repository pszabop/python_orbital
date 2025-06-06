FROM python:3.6.1

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install numpy scipy matplotlib
