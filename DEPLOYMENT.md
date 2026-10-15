# Production Deployment Guide — Design Matcher Licensing Server

This guide provides step-by-step instructions for deploying the **Design Matcher Licensing Server** in a secure, production-grade Linux environment using Docker Compose or bare-metal/systemd with PostgreSQL and an HTTPS reverse proxy.

---

## 1. Prerequisites & System Requirements

- **Operating System:** Ubuntu 22.04 LTS, Ubuntu 24.04 LTS, Debian 12, or AlmaLinux 9
- **Hardware Minimum:** 2 vCPU, 4 GB RAM, 20 GB SSD storage
- **Software:**
  - Docker & Docker Compose (v2.20+) OR Python 3.11+ / PostgreSQL 14+
  - OpenSSL 3.0+
  - Public Domain Name (e.g., `licensing.yourcompany.com`) with DNS A record pointing to your server IP.

---

## 2. Generate Production Ed25519 Signing Keys

> [!CAUTION]
> The private key signs licenses. If compromised, unauthorized licenses can be forged. Store the private key exclusively on the secure server or in a secret manager (AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault). Never commit the private key to source control!

Run the following commands on your production server to generate the keypair:

```bash
# 1. Generate the Ed25519 Private Key in PEM format
openssl genpkey -algorithm ed25519 -out license_private_key.pem

# 2. Extract the corresponding Public Key in PEM format
openssl pkey -in license_private_key.pem -pubout -out license_public_key.pem

# 3. Secure file permissions (Owner read-only)
chmod 600 license_private_key.pem
chmod 644 license_public_key.pem
```

The contents of `license_public_key.pem` are what you embed into the client app (`design_matcher/app/core/licensing/public_key.py`).

---

## 3. Option A: Deployment with Docker Compose (Recommended)

### Directory Structure
```
/opt/licensing_server/
├── docker-compose.yml
├── .env
├── nginx/
│   └── default.conf
└── certs/
    ├── license_private_key.pem
    └── license_public_key.pem
```

### 3.1. Create `.env` Configuration File
```bash
# PostgreSQL Settings
POSTGRES_DB=licensing_prod
POSTGRES_USER=licensing_user
POSTGRES_PASSWORD=GENERATE_STRONG_DB_PASSWORD_HERE

# Database Connection URL for FastAPI
DATABASE_URL=postgresql+psycopg2://licensing_user:GENERATE_STRONG_DB_PASSWORD_HERE@postgres:5432/licensing_prod

# Ed25519 Key Paths
LICENSE_PRIVATE_KEY_PATH=/certs/license_private_key.pem
LICENSE_PUBLIC_KEY_PATH=/certs/license_public_key.pem

# Security & Admin Access
JWT_SECRET=GENERATE_64_CHAR_HEX_SECRET_HERE
JWT_EXPIRE_MINUTES=1440
ADMIN_DEFAULT_USER=admin
ADMIN_DEFAULT_PASSWORD=GENERATE_SUPER_SECURE_ADMIN_PASSWORD_HERE

# Server Settings
PORT=8000
ENVIRONMENT=production
CORS_ORIGINS=["https://licensing.yourcompany.com"]
```

### 3.2. Dockerfile (`licensing_server/Dockerfile`)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt psycopg2-binary

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "alembic -c alembic.ini upgrade head && uvicorn licensing_server.main:app --host 0.0.0.0 --port 8000 --workers 4"]
```

### 3.3. `docker-compose.yml`
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: licensing_postgres
    restart: always
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - internal_net

  api:
    build:
      context: .
      dockerfile: licensing_server/Dockerfile
    container_name: licensing_api
    restart: always
    depends_on:
      - postgres
    env_file:
      - .env
    volumes:
      - ./certs:/certs:ro
    networks:
      - internal_net
    expose:
      - "8000"

  nginx:
    image: nginx:alpine
    container_name: licensing_nginx
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/default.conf:/etc/nginx/conf.d/default.conf:ro
      - /etc/letsencrypt:/etc/letsencrypt:ro
    depends_on:
      - api
    networks:
      - internal_net

volumes:
  postgres_data:

networks:
  internal_net:
    driver: bridge
```

### 3.4. Nginx Reverse Proxy Configuration (`nginx/default.conf`)
```nginx
# Rate limiting zone for public license checks
limit_req_zone $binary_remote_addr zone=licensing_limit:10m rate=15r/m;

server {
    listen 80;
    server_name licensing.yourcompany.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name licensing.yourcompany.com;

    ssl_certificate /etc/letsencrypt/live/licensing.yourcompany.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/licensing.yourcompany.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security Headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Rate-limited Activation & Validation
    location /api/v1/licenses/ {
        limit_req zone=licensing_limit burst=5 nodelay;
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Admin Portal & Other API Endpoints
    location / {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 4. Option B: Deployment with Bare-Metal Systemd & Caddy

If you prefer deploying directly on Ubuntu without Docker:

### 4.1. Systemd Service File (`/etc/systemd/system/licensing.service`)
```ini
[Unit]
Description=Design Matcher Licensing Server
After=network.target postgresql.service

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/licensing_server
EnvironmentFile=/opt/licensing_server/.env
ExecStart=/opt/licensing_server/.venv/bin/uvicorn licensing_server.main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable licensing
sudo systemctl start licensing
```

### 4.2. Caddyfile (`/etc/caddy/Caddyfile`)
Caddy automatically handles Let's Encrypt SSL certificates:
```caddy
licensing.yourcompany.com {
    reverse_proxy 127.0.0.1:8000
    encode gzip zstd
}
```

---

## 5. Database Migrations

Always run database migrations upon initial deployment and subsequent updates:
```bash
# Run migrations using Alembic
alembic -c licensing_server/alembic.ini upgrade head
```

---

## 6. Database Backups

Set up an automated daily backup cron job for PostgreSQL:

```bash
# /etc/cron.daily/backup-licensing-db
#!/bin/bash
BACKUP_DIR="/var/backups/licensing"
mkdir -p "$BACKUP_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
pg_dump -U licensing_user -h localhost licensing_prod | gzip > "$BACKUP_DIR/licensing_db_$TIMESTAMP.sql.gz"
find "$BACKUP_DIR" -type f -name "*.sql.gz" -mtime +30 -delete
```
chmod +x `/etc/cron.daily/backup-licensing-db`.

---

## 7. Production Verification Checklist

1. [ ] **TLS Certificate:** Ensure `https://licensing.yourcompany.com` resolves with a valid SSL/TLS certificate.
2. [ ] **Health Check:** `curl -I https://licensing.yourcompany.com/api/v1/licenses/status/test` returns `404` (not connection refused).
3. [ ] **Admin Portal:** Visit `https://licensing.yourcompany.com/admin` and log in with the administrator credentials.
4. [ ] **Public Key Match:** Verify that the public key configured on the server matches `design_matcher/app/core/licensing/public_key.py`.
5. [ ] **Private Key Secrecy:** Verify that `license_private_key.pem` is NOT included in the desktop client distribution files or PyInstaller build bundle.
