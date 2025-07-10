#!/usr/bin/env python3
"""
Quick DMA Activity Check
Monitors descriptor advancement to see if DMA is actively streaming
"""

import time
from koheron import connect, command

class CurrentRamp:
    def __init__(self, client):
        self.client = client

    @command()
    def get_current_descriptor_index(self):
        return self.client.recv_uint32()
    
    @command()
    def get_dma_status_register(self):
        return self.client.recv_uint32()
    
    @command()
    def get_dma_running(self):
        return self.client.recv_bool()
    
    @command()
    def get_dma_idle(self):
        return self.client.recv_bool()
    
    @command()
    def get_dma_error(self):
        return self.client.recv_bool()
    
    @command()
    def is_adc_streaming_active(self):
        return self.client.recv_bool()
    
    @command()
    def start_adc_streaming(self):
        pass
    
    @command()
    def stop_adc_streaming(self):
        pass
    
    @command()
    def set_cic_decimation_rate(self, rate):
        pass

def decode_dma_status(status_reg):
    """Decode DMA status register bits"""
    print(f"    Raw status: 0x{status_reg:08x}")
    print(f"    Bit breakdown:")
    print(f"      RS (Running):        {bool(status_reg & 0x1)}")
    print(f"      Idle:                {bool(status_reg & 0x2)}")
    print(f"      SGIncld:             {bool(status_reg & 0x8)}")
    print(f"      DMAIntErr:           {bool(status_reg & 0x10)}")
    print(f"      DMASlvErr:           {bool(status_reg & 0x20)}")
    print(f"      DMADecErr:           {bool(status_reg & 0x40)}")
    print(f"      SGIntErr:            {bool(status_reg & 0x100)}")
    print(f"      SGSlvErr:            {bool(status_reg & 0x200)}")
    print(f"      SGDecErr:            {bool(status_reg & 0x400)}")
    print(f"      IOC_Irq:             {bool(status_reg & 0x1000)}")
    print(f"      Dly_Irq:             {bool(status_reg & 0x2000)}")
    print(f"      Err_Irq:             {bool(status_reg & 0x4000)}")

def main():
    print("🔍 Enhanced DMA Activity Check")
    print("=" * 50)
    
    try:
        client = connect('192.168.1.115', 'alpha15-laser-control', restart=False)
        driver = CurrentRamp(client)
        
        # Check detailed DMA status
        print("📊 Initial DMA Status:")
        status_reg = driver.get_dma_status_register()
        decode_dma_status(status_reg)
        
        print(f"\n📊 Driver Status:")
        active = driver.is_adc_streaming_active()
        dma_running = driver.get_dma_running()
        dma_idle = driver.get_dma_idle()
        dma_error = driver.get_dma_error()
        
        print(f"    Streaming active: {active}")
        print(f"    DMA running:      {dma_running}")
        print(f"    DMA idle:         {dma_idle}")
        print(f"    DMA error:        {dma_error}")
        
        if not active:
            print("\n🔧 Restarting streaming...")
            driver.stop_adc_streaming()
            time.sleep(1.0)
            driver.set_cic_decimation_rate(2500)
            driver.start_adc_streaming()
            time.sleep(3.0)
            
            print("📊 Status After Restart:")
            status_reg = driver.get_dma_status_register()
            decode_dma_status(status_reg)
            
            active = driver.is_adc_streaming_active()
            print(f"    Streaming active: {active}")
        
        # Monitor descriptor index for 20 seconds
        print(f"\n🔍 Monitoring descriptor advancement (20s):")
        start_time = time.time()
        last_desc = -1
        changes = 0
        
        for i in range(20):  # 20 seconds
            try:
                desc_idx = driver.get_current_descriptor_index()
                elapsed = time.time() - start_time
                
                if desc_idx != last_desc and last_desc >= 0:
                    changes += 1
                    print(f"  {elapsed:5.1f}s: Desc {last_desc} → {desc_idx} (change #{changes})")
                elif i % 5 == 0:  # Print every 5 seconds
                    print(f"  {elapsed:5.1f}s: Desc {desc_idx} (no change)")
                
                last_desc = desc_idx
                time.sleep(1.0)
                
            except Exception as e:
                print(f"  Error reading descriptor: {e}")
                break
        
        # Final status check
        print(f"\n📊 Final Status Check:")
        status_reg = driver.get_dma_status_register()
        decode_dma_status(status_reg)
        
        print(f"\n🎯 Result: {changes} descriptor changes in 20 seconds")
        
        if changes > 0:
            print("✅ DMA is actively advancing - streaming is working!")
        else:
            print("❌ DMA descriptors not advancing")
            print("💡 Possible causes:")
            print("   - CIC decimator not producing data")
            print("   - Clock domain crossing issue")
            print("   - DMA not properly started")
            print("   - TLAST generator not working")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 