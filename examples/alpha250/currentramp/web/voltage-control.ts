// Current Ramp Control driver
// (c) Koheron

class VoltageControl {
    private driver: Driver;
    private id: number;
    private cmds: Commands;

    constructor(private client: Client) {
        this.driver = this.client.getDriver('CurrentRamp');
        this.id = this.driver.id;
        this.cmds = this.driver.getCmds();
    }

    // === DC Temperature Control Functions ===
    
    setTemperatureDcVoltage(voltage: number): void {
        this.client.send(Command(this.id, this.cmds['set_temperature_dc_voltage'], voltage));
    }

    enableTemperatureDcOutput(enable: boolean): void {
        this.client.send(Command(this.id, this.cmds['enable_temperature_dc_output'], enable));
    }

    getTemperatureDcVoltage(callback: (voltage: number) => void): void {
        this.client.readFloat32(Command(this.id, this.cmds['get_temperature_dc_voltage']), callback);
    }

    getTemperatureDcEnabled(callback: (enabled: boolean) => void): void {
        this.client.readBool(Command(this.id, this.cmds['get_temperature_dc_enabled']), callback);
    }

    // === Current Ramp Control Functions ===
    
    setRampOffset(offset: number): void {
        this.client.send(Command(this.id, this.cmds['set_ramp_offset'], offset));
    }

    setRampAmplitude(amplitude: number): void {
        this.client.send(Command(this.id, this.cmds['set_ramp_amplitude'], amplitude));
    }

    setRampFrequency(frequency: number): void {
        this.client.send(Command(this.id, this.cmds['set_ramp_frequency'], frequency));
    }

    startRamp(): void {
        this.client.send(Command(this.id, this.cmds['start_ramp']));
    }

    stopRamp(): void {
        this.client.send(Command(this.id, this.cmds['stop_ramp']));
    }

    generateRampWaveform(): void {
        this.client.send(Command(this.id, this.cmds['generate_ramp_waveform']));
    }

    getRampOffset(callback: (offset: number) => void): void {
        this.client.readFloat32(Command(this.id, this.cmds['get_ramp_offset']), callback);
    }

    getRampAmplitude(callback: (amplitude: number) => void): void {
        this.client.readFloat32(Command(this.id, this.cmds['get_ramp_amplitude']), callback);
    }

    // Legacy functions for backward compatibility
    setVoltageOutput(voltage: number): void {
        this.setTemperatureDcVoltage(voltage);
    }

    enableOutput(enable: boolean): void {
        this.enableTemperatureDcOutput(enable);
    }
}

class VoltageControlApp {

    // DC Control Elements
    private dcOutputToggle: HTMLInputElement;
    private dcVoltageInput: HTMLInputElement;
    private dcVoltageDisplay: HTMLSpanElement;
    private dcStatusDisplay: HTMLSpanElement;
    private setDcVoltageBtn: HTMLButtonElement;

    // Ramp Control Elements
    private rampEnableToggle: HTMLInputElement;
    private rampStatusDisplay: HTMLSpanElement;
    
    private frequencyInput: HTMLInputElement;
    private amplitudeInput: HTMLInputElement;
    private offsetInput: HTMLInputElement;
    
    private setFrequencyBtn: HTMLButtonElement;
    private setAmplitudeBtn: HTMLButtonElement;
    private setOffsetBtn: HTMLButtonElement;
    
    private currentFrequencySpan: HTMLSpanElement;
    private currentAmplitudeSpan: HTMLSpanElement;
    private currentOffsetSpan: HTMLSpanElement;

    constructor(document: Document, private voltageControl: VoltageControl) {
        // DC Control Elements
        this.dcOutputToggle = <HTMLInputElement>document.getElementById('dc-output-toggle');
        this.dcVoltageInput = <HTMLInputElement>document.getElementById('dc-voltage-input');
        this.dcVoltageDisplay = <HTMLSpanElement>document.getElementById('dc-voltage-display');
        this.dcStatusDisplay = <HTMLSpanElement>document.getElementById('dc-status-display');
        this.setDcVoltageBtn = <HTMLButtonElement>document.getElementById('set-dc-voltage');

        // Ramp Control Elements
        this.rampEnableToggle = <HTMLInputElement>document.getElementById('ramp-enable-toggle');
        this.rampStatusDisplay = <HTMLSpanElement>document.getElementById('ramp-status-display');
        
        this.frequencyInput = <HTMLInputElement>document.getElementById('frequency-input');
        this.amplitudeInput = <HTMLInputElement>document.getElementById('amplitude-input');
        this.offsetInput = <HTMLInputElement>document.getElementById('offset-input');
        
        this.setFrequencyBtn = <HTMLButtonElement>document.getElementById('set-frequency');
        this.setAmplitudeBtn = <HTMLButtonElement>document.getElementById('set-amplitude');
        this.setOffsetBtn = <HTMLButtonElement>document.getElementById('set-offset');
        
        this.currentFrequencySpan = <HTMLSpanElement>document.getElementById('current-frequency');
        this.currentAmplitudeSpan = <HTMLSpanElement>document.getElementById('current-amplitude');
        this.currentOffsetSpan = <HTMLSpanElement>document.getElementById('current-offset');

        // Check for missing elements - prob don't need this anymore
        const requiredElements = [
            this.dcOutputToggle, this.dcVoltageInput, this.dcVoltageDisplay, this.dcStatusDisplay, this.setDcVoltageBtn,
            this.rampEnableToggle, this.rampStatusDisplay, this.frequencyInput, this.amplitudeInput, this.offsetInput,
            this.setFrequencyBtn, this.setAmplitudeBtn, this.setOffsetBtn, 
            this.currentFrequencySpan, this.currentAmplitudeSpan, this.currentOffsetSpan
        ];
        
        for (let i = 0; i < requiredElements.length; i++) {
            if (!requiredElements[i]) {
                console.error(`Missing DOM element at index ${i}. Check voltage-control.html template.`);
                return; // Don't proceed if elements are missing
            }
        }

        this.setupEventListeners();
        this.updateStatus();
    }

    private setupEventListeners(): void {
        // DC Control Event Listeners
        this.dcOutputToggle.addEventListener('change', (event) => {
            const enabled = (<HTMLInputElement>event.target).checked;
            this.voltageControl.enableTemperatureDcOutput(enabled);
            setTimeout(() => this.updateStatus(), 100);
        });

        this.setDcVoltageBtn.addEventListener('click', () => {
            const voltage = parseFloat(this.dcVoltageInput.value);
            if (!isNaN(voltage) && voltage >= 0 && voltage <= 2.5) {
                this.voltageControl.setTemperatureDcVoltage(voltage);
                setTimeout(() => this.updateStatus(), 100);
            } else {
                alert('Please enter a valid voltage between 0 and 2.5V');
            }
        });

        // Ramp Control Event Listeners
        this.rampEnableToggle.addEventListener('change', (event) => {
            const enabled = (<HTMLInputElement>event.target).checked;
            if (enabled) {
                this.voltageControl.startRamp();
                this.rampStatusDisplay.textContent = 'Running';
                this.rampStatusDisplay.className = 'label label-success';
            } else {
                this.voltageControl.stopRamp();
                this.rampStatusDisplay.textContent = 'Stopped';
                this.rampStatusDisplay.className = 'label label-default';
            }
        });

        this.setFrequencyBtn.addEventListener('click', () => {
            const frequency = parseFloat(this.frequencyInput.value);
            if (!isNaN(frequency) && frequency >= 0.1 && frequency <= 1000) {
                this.voltageControl.setRampFrequency(frequency);
                this.currentFrequencySpan.textContent = frequency.toFixed(1);
            } else {
                alert('Please enter a valid frequency between 0.1 and 1000 Hz');
            }
        });

        this.setAmplitudeBtn.addEventListener('click', () => {
            const amplitude = parseFloat(this.amplitudeInput.value);
            const offset = parseFloat(this.offsetInput.value);
            
            if (!isNaN(amplitude) && amplitude >= 0 && amplitude <= 2.5) {
                if (amplitude + offset <= 2.5) {
                    this.voltageControl.setRampAmplitude(amplitude);
                    this.currentAmplitudeSpan.textContent = amplitude.toFixed(2);
                    this.voltageControl.generateRampWaveform(); // Regenerate waveform with new amplitude
                } else {
                    alert('Amplitude + Offset must not exceed 2.5V');
                }
            } else {
                alert('Please enter a valid amplitude between 0 and 2.5V');
            }
        });

        this.setOffsetBtn.addEventListener('click', () => {
            const offset = parseFloat(this.offsetInput.value);
            const amplitude = parseFloat(this.amplitudeInput.value);
            
            if (!isNaN(offset) && offset >= 0 && offset <= 2.5) {
                if (amplitude + offset <= 2.5) {
                    this.voltageControl.setRampOffset(offset);
                    this.currentOffsetSpan.textContent = offset.toFixed(2);
                    this.voltageControl.generateRampWaveform(); // Regenerate waveform with new offset
                } else {
                    alert('Amplitude + Offset must not exceed 2.5V');
                }
            } else {
                alert('Please enter a valid offset between 0 and 2.5V');
            }
        });

        // Allow Enter key to trigger set buttons
        this.dcVoltageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.setDcVoltageBtn.click();
        });
        this.frequencyInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.setFrequencyBtn.click();
        });
        this.amplitudeInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.setAmplitudeBtn.click();
        });
        this.offsetInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.setOffsetBtn.click();
        });
    }

    private updateStatus(): void {
        // Update DC Control Status
        this.voltageControl.getTemperatureDcEnabled((enabled) => {
            this.dcOutputToggle.checked = enabled;
            this.dcStatusDisplay.textContent = enabled ? 'Enabled' : 'Disabled';
            this.dcStatusDisplay.className = enabled ? 'label label-success' : 'label label-default';

            if (enabled) {
                this.voltageControl.getTemperatureDcVoltage((voltage) => {
                    this.dcVoltageDisplay.textContent = voltage.toFixed(3);
                });
            } else {
                this.dcVoltageDisplay.textContent = '0.000';
            }
        });

        // Update Current Ramp Settings
        this.voltageControl.getRampOffset((offset) => {
            this.currentOffsetSpan.textContent = offset.toFixed(2);
        });
        
        this.voltageControl.getRampAmplitude((amplitude) => {
            this.currentAmplitudeSpan.textContent = amplitude.toFixed(2);
        });

        // Update status every 1000ms
        setTimeout(() => this.updateStatus(), 1000);
    }
} 