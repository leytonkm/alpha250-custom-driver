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
        
        // Debug: Log available commands
        console.log('Available CurrentRamp commands:', Object.keys(this.cmds));
    }

    // === DC Temperature Control Functions ===
    
    setTemperatureDcVoltage(voltage: number): void {
        console.log('Setting DC voltage to:', voltage);
        this.client.send(Command(this.id, this.cmds['set_temperature_dc_voltage'], voltage));
    }

    enableTemperatureDcOutput(enable: boolean): void {
        console.log('Enabling DC output:', enable);
        this.client.send(Command(this.id, this.cmds['enable_temperature_dc_output'], enable));
    }

    getTemperatureDcVoltage(callback: (voltage: number) => void): void {
        this.client.readFloat32(Command(this.id, this.cmds['get_temperature_dc_voltage']), (voltage) => {
            console.log('Read DC voltage:', voltage);
            callback(voltage);
        });
    }

    getTemperatureDcEnabled(callback: (enabled: boolean) => void): void {
        this.client.readBool(Command(this.id, this.cmds['get_temperature_dc_enabled']), (enabled) => {
            console.log('Read DC enabled:', enabled);
            callback(enabled);
        });
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

    // Default values for ramp parameters
    private static readonly DEFAULT_FREQUENCY = 10.0;
    private static readonly DEFAULT_AMPLITUDE = 1.0;
    private static readonly DEFAULT_OFFSET = 1.5;

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
    private resetDefaultsBtn: HTMLButtonElement;
    
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
        this.resetDefaultsBtn = <HTMLButtonElement>document.getElementById('reset-defaults');
        
        this.currentFrequencySpan = <HTMLSpanElement>document.getElementById('current-frequency');
        this.currentAmplitudeSpan = <HTMLSpanElement>document.getElementById('current-amplitude');
        this.currentOffsetSpan = <HTMLSpanElement>document.getElementById('current-offset');

        // Check for missing elements - prob don't need this anymore
        const requiredElements = [
            this.dcOutputToggle, this.dcVoltageInput, this.dcVoltageDisplay, this.dcStatusDisplay, this.setDcVoltageBtn,
            this.rampEnableToggle, this.rampStatusDisplay, this.frequencyInput, this.amplitudeInput, this.offsetInput,
            this.setFrequencyBtn, this.setAmplitudeBtn, this.setOffsetBtn, 
            this.currentFrequencySpan, this.currentAmplitudeSpan, this.currentOffsetSpan, this.resetDefaultsBtn
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
            if (enabled) {
                // When enabling, first set the voltage then enable output
                const voltage = parseFloat(this.dcVoltageInput.value);
                if (!isNaN(voltage) && voltage >= 0 && voltage <= 2.5) {
                    // Update UI immediately for responsiveness
                    this.dcStatusDisplay.textContent = 'Enabled';
                    this.dcStatusDisplay.className = 'label label-success';
                    
                    this.voltageControl.setTemperatureDcVoltage(voltage);
                    // Shorter delay between commands
                    setTimeout(() => {
                        this.voltageControl.enableTemperatureDcOutput(enabled);
                        // Shorter delay before status check
                        setTimeout(() => this.updateStatus(), 150);
                    }, 50);
                } else {
                    // If voltage is invalid, don't enable and show error
                    this.dcOutputToggle.checked = false;
                    alert('Please set a valid voltage between 0 and 2.5V before enabling');
                    return;
                }
            } else {
                // When disabling, update UI immediately and disable quickly
                this.dcStatusDisplay.textContent = 'Disabled';
                this.dcStatusDisplay.className = 'label label-default';
                this.voltageControl.enableTemperatureDcOutput(enabled);
                setTimeout(() => this.updateStatus(), 100);
            }
        });

        this.setDcVoltageBtn.addEventListener('click', () => {
            const voltage = parseFloat(this.dcVoltageInput.value);
            if (!isNaN(voltage) && voltage >= 0 && voltage <= 2.5) {
                this.voltageControl.setTemperatureDcVoltage(voltage);
                // Immediately update display if output is enabled
                if (this.dcOutputToggle.checked) {
                    this.dcVoltageDisplay.textContent = voltage.toFixed(3);
                }
                // Wait a bit longer for the backend to process
                setTimeout(() => this.updateStatus(), 200);
            } else {
                alert('Please enter a valid voltage between 0 and 2.5V');
            }
        });

        // Ramp Control Event Listeners
        this.rampEnableToggle.addEventListener('change', (event) => {
            const enabled = (<HTMLInputElement>event.target).checked;
            if (enabled) {
                this.voltageControl.startRamp();
                this.rampStatusDisplay.textContent = 'Enabled';
                this.rampStatusDisplay.className = 'label label-success';
            } else {
                this.voltageControl.stopRamp();
                this.rampStatusDisplay.textContent = 'Disabled';
                this.rampStatusDisplay.className = 'label label-default';
            }
        });

        this.setFrequencyBtn.addEventListener('click', () => {
            const frequency = parseFloat(this.frequencyInput.value);
            if (!isNaN(frequency) && frequency >= 0.1 && frequency <= 1000000) {
                const rampWasRunning = this.rampEnableToggle.checked;
                
                // Stop ramp if running
                if (rampWasRunning) {
                    this.voltageControl.stopRamp();
                }
                
                // Set new frequency and regenerate waveform
                this.voltageControl.setRampFrequency(frequency);
                this.currentFrequencySpan.textContent = frequency.toFixed(1);
                this.voltageControl.generateRampWaveform();
                
                // Restart ramp if it was running
                if (rampWasRunning) {
                    this.voltageControl.startRamp();
                }
            } else {
                alert('Please enter a valid frequency between 0.1 and 1000000 Hz');
            }
        });

        this.setAmplitudeBtn.addEventListener('click', () => {
            const amplitude = parseFloat(this.amplitudeInput.value);
            const offset = parseFloat(this.offsetInput.value);
            
            if (!isNaN(amplitude) && amplitude >= 0 && amplitude <= 2.5) {
                if (amplitude + offset <= 2.5) {
                    const rampWasRunning = this.rampEnableToggle.checked;
                    
                    // Stop ramp if running
                    if (rampWasRunning) {
                        this.voltageControl.stopRamp();
                    }
                    
                    // Set new amplitude and regenerate waveform
                    this.voltageControl.setRampAmplitude(amplitude);
                    this.currentAmplitudeSpan.textContent = amplitude.toFixed(2);
                    this.voltageControl.generateRampWaveform();
                    
                    // Restart ramp if it was running
                    if (rampWasRunning) {
                        this.voltageControl.startRamp();
                    }
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
                    const rampWasRunning = this.rampEnableToggle.checked;
                    
                    // Stop ramp if running
                    if (rampWasRunning) {
                        this.voltageControl.stopRamp();
                    }
                    
                    // Set new offset and regenerate waveform
                    this.voltageControl.setRampOffset(offset);
                    this.currentOffsetSpan.textContent = offset.toFixed(2);
                    this.voltageControl.generateRampWaveform();
                    
                    // Restart ramp if it was running
                    if (rampWasRunning) {
                        this.voltageControl.startRamp();
                    }
                } else {
                    alert('Amplitude + Offset must not exceed 2.5V');
                }
            } else {
                alert('Please enter a valid offset between 0 and 2.5V');
            }
        });

        this.resetDefaultsBtn.addEventListener('click', () => {
            this.frequencyInput.value = VoltageControlApp.DEFAULT_FREQUENCY.toFixed(1);
            this.amplitudeInput.value = VoltageControlApp.DEFAULT_AMPLITUDE.toFixed(2);
            this.offsetInput.value = VoltageControlApp.DEFAULT_OFFSET.toFixed(2);
            this.setFrequencyBtn.click();
            this.setAmplitudeBtn.click();
            this.setOffsetBtn.click();
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