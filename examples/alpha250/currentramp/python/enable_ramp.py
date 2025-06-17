#!/usr/bin/env python3
"""
Enable Ramp for Web Graphing Test
"""

import os
from koheron import connect, command

class CurrentRamp:
    def __init__(self, client):
        self.client = client

    @command()
    def generate_ramp_waveform(self):
        """Generate ramp waveform"""
        pass

    @command()
    def start_ramp(self):
        """Start the ramp"""
        pass

    @command()
    def get_ramp_enabled(self):
        """Check if ramp is enabled"""
        return self.client.recv_bool()

def enable_ramp():
    """Enable the ramp for testing"""
    print("🚀 Enabling Ramp for Web Graphing Test")
    
    try:
        host = os.getenv('HOST', '192.168.1.20')
        client = connect(host, 'currentramp', restart=False)
        driver = CurrentRamp(client)
        print(f"✅ Connected to {host}")
        
        # Generate waveform and start ramp
        print("📡 Generating ramp waveform...")
        driver.generate_ramp_waveform()
        
        print("▶️  Starting ramp...")
        driver.start_ramp()
        
        # Verify it's enabled
        import time
        time.sleep(0.5)  # Wait a bit
        enabled = driver.get_ramp_enabled()
        
        if enabled:
            print("✅ Ramp is now ENABLED!")
            print("🎯 Now try the web graphing - you should see data!")
        else:
            print("❌ Ramp failed to enable")
            
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == '__main__':
    enable_ramp() 