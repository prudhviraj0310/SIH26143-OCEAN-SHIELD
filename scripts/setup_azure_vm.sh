#!/usr/bin/env bash
# ==============================================================================
# OCEAN-SHIELD — One-Click Azure VM Deployment Script
# Target: Azure Ubuntu VM (sih265)
# ==============================================================================

set -e

# Prefer oceanshield_key.pem if present, else fallback to sih265_key.pem
VM_IP="$1"
DOMAIN="${2:-_}"
if [ -n "$3" ]; then
    KEY_FILE="$3"
elif [ -f "/Users/prudhviraj/Downloads/planning for sih/oceanshield_key.pem" ]; then
    KEY_FILE="/Users/prudhviraj/Downloads/planning for sih/oceanshield_key.pem"
else
    KEY_FILE="/Users/prudhviraj/Downloads/planning for sih/sih265_key.pem"
fi

USER="azureuser"

if [ -z "$1" ]; then
    echo "❌ Usage: ./scripts/setup_azure_vm.sh <AZURE_VM_PUBLIC_IP> [DOMAIN] [KEY_FILE]"
    echo "Example: ./scripts/setup_azure_vm.sh 20.198.54.12 mydomain.com"
    exit 1
fi

echo "🔐 Using SSH key: $KEY_FILE"
chmod 400 "$KEY_FILE"

echo "🚀 Connecting to Azure VM ($USER@$VM_IP)..."

ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no "$USER@$VM_IP" bash -s -- "$DOMAIN" << 'EOF'
set -e

DOMAIN="${1:-_}"

echo "📦 1/6 Updating system packages & installing Nginx..."
sudo apt update -y
sudo apt install -y python3-pip python3-venv git curl htop nginx

echo "📂 2/6 Setting up project repository..."
cd ~
if [ ! -d "SIH26143-OCEAN-SHIELD" ]; then
    git clone https://github.com/prudhviraj0310/SIH26143-OCEAN-SHIELD.git
    cd SIH26143-OCEAN-SHIELD
else
    cd SIH26143-OCEAN-SHIELD
    git pull origin main || true
fi

echo "🐍 3/6 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo "⚡ 4/6 Installing PyTorch CPU & dependencies..."
pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r requirements.txt

echo "🔄 5/6 Configuring 24/7 background systemd service..."
sudo tee /etc/systemd/system/oceanshield.service > /dev/null << 'SERVICE'
[Unit]
Description=OCEAN-SHIELD 24/7 AI Maritime Backend
After=network.target

[Service]
User=azureuser
WorkingDirectory=/home/azureuser/SIH26143-OCEAN-SHIELD
ExecStart=/home/azureuser/SIH26143-OCEAN-SHIELD/venv/bin/uvicorn src.ocean_shield.server:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable oceanshield
sudo systemctl restart oceanshield

echo "🌐 6/6 Configuring Nginx Reverse Proxy with WebSocket & Cloudflare support..."
sudo tee /etc/nginx/sites-available/oceanshield > /dev/null << NGINX
server {
    listen 80;
    server_name $DOMAIN;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
    }
}
NGINX

sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -sf /etc/nginx/sites-available/oceanshield /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx

echo ""
echo "=========================================================="
echo "✅ OCEAN-SHIELD is now running 24/7 behind Nginx on your Azure VM!"
echo "Service Status:"
sudo systemctl status oceanshield --no-pager
echo "Nginx Status:"
sudo systemctl status nginx --no-pager
echo "=========================================================="
EOF

echo ""
echo "🎉 SUCCESS! Your live application is accessible at:"
echo "👉 http://$VM_IP"
if [ "$DOMAIN" != "_" ]; then
    echo "👉 http://$DOMAIN (via Cloudflare once DNS is pointed)"
fi

