# -------------------------
# Base Image
# -------------------------
FROM python:3.11-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# -------------------------
# System deps for Playwright
# -------------------------
RUN apt-get update && apt-get install -y \
    wget gnupg unzip curl \
    libnss3 libnspr4 libx11-6 libx11-xcb1 libxcomposite1 libxdamage1 \
    libxrandr2 libxext6 libxfixes3 libxi6 libxrender1 libxss1 libglib2.0-0 \
    libdbus-1-3 libxtst6 libxshmfence1 xvfb fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Install Python deps
# -------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Install Playwright Browsers
# -------------------------
RUN playwright install --with-deps chromium firefox

# -------------------------
# Copy project files
# -------------------------
COPY . .

# -------------------------
# Expose port & start FastAPI
# -------------------------
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
