#!/usr/bin/env bash
# ==============================================================================
# OCEAN-SHIELD — One-Click Azure VM Deployment Script
# Target: Azure Ubuntu VM (sih265)
# ==============================================================================

set -e

# Prefer oceanshield_key.pem if present, else fallback to sih265_key.pem
if [ -n "$2" ]; then
    KEY_FILE="$2"
elif [ -f "/Users/prudhviraj/Downloads/planning for sih/oceanshield_key.pem" ]; then
    KEY_FILE="/Users/prudhviraj/Downloads/planning for sih/oceanshield_key.pem"
else
    KEY_FILE="/Users/prudhviraj/Downloads/planning for sih/sih265_key.pem"
fi

USER="azureuser"

if [ -z "$1" ]; then
    echo "❌ Usage: ./scripts/setup_azure_vm.sh <AZURE_VM_PUBLIC_IP> [KEY_FILE]"
    echo "Example: ./scripts/setup_azure_vm.sh 20.198.54.12"
    exit 1
fi

VM_IP="$1"

echo "🔐 Using SSH key: $KEY_FILE"
chmod 400 "$KEY_FILE"

echo "🚀 Connecting to Azure VM ($USER@$VM_IP)..."

ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no "$USER@$VM_IP" bash -s << 'EOF'
set -e

echo "📦 1/5 Updating system packages..."
sudo apt update -y
sudo apt install -y python3-pip python3-venv git curl htop

echo "📂 2/5 Setting up project repository..."
cd ~
if [ ! -d "SIH26143-OCEAN-SHIELD" ]; then
    git clone https://github.com/prudhviraj0310/SIH26143-OCEAN-SHIELD.git
    cd SIH26143-OCEAN-SHIELD
else
    cd SIH26143-OCEAN-SHIELD
    git pull origin main || true
fi

echo "🐍 3/5 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo "⚡ 4/5 Installing PyTorch CPU & dependencies..."
pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r requirements.txt

echo "🔄 5/5 Configuring 24/7 background systemd service..."
sudo tee /etc/systemd/system/oceanshield.service > /dev/null << 'SERVICE'
[Unit]
Description=OCEAN-SHIELD 24/7 AI Maritime Backend
After=network.target

[Service]
User=azureuser
WorkingDirectory=/home/azureuser/SIH26143-OCEAN-SHIELD
ExecStart=/home/azureuser/SIH26143-OCEAN-SHIELD/venv/bin/uvicorn src.ocean_shield.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
SERVICE

# Port redirect so port 80 maps directly to 8000
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8000 || true
sudo iptables -t nat -A OUTPUT -p tcp -o lo --dport 80 -j REDIRECT --to-port 8000 || true
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent && sudo netfilter-persistent save || true

sudo systemctl daemon-reload
sudo systemctl enable oceanshield
sudo systemctl restart oceanshield

echo ""
echo "=========================================================="
echo "✅ OCEAN-SHIELD is now running 24/7 on your Azure VM!"
echo "Status:"
sudo systemctl status oceanshield --no-pager
echo "=========================================================="
EOF

echo ""
echo "🎉 SUCCESS! Your live application is accessible at:"
echo "👉 http://$VM_IP:8000"
echo "👉 http://$VM_IP"
