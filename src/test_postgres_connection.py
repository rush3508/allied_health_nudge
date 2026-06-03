from sqlalchemy import create_engine, text

# Tailscale (preferred) — V530s Tailscale IP
V530S_IP = 'localhost'

# Local LAN fallback — uncomment and replace with output of `hostname -I` on V530s
# V530S_IP = '192.168.x.x'

engine = create_engine(f"postgresql://ds_user:choose_a_password@{V530S_IP}:5432/allied_health")

with engine.connect() as conn:
    result = conn.execute(text("SELECT version()"))
    print("Connected:", result.fetchone()[0])