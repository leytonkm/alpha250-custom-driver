// Voltage Control driver
// (c) Koheron

class VoltageControl {
    private driver: Driver;
    private id: number;
    private cmds: Commands;

    constructor(private client: Client) {
        this.driver = this.client.getDriver('VoltageControl');
        this.id = this.driver.id;
        this.cmds = this.driver.getCmds();
    }

    setVoltageOutput(voltage: number): void {
        this.client.send(Command(this.id, this.cmds['set_voltage_output'], voltage));
    }

    enableOutput(enable: boolean): void {
        this.client.send(Command(this.id, this.cmds['enable_output'], enable));
    }

    toggleOutput(): void {
        this.client.send(Command(this.id, this.cmds['toggle_output']));
    }

    setTestVoltage(): void {
        this.client.send(Command(this.id, this.cmds['set_test_voltage']));
    }

    disableTestVoltage(): void {
        this.client.send(Command(this.id, this.cmds['disable_test_voltage']));
    }

    getOutputVoltage(callback: (voltage: number) => void): void {
        this.client.readFloat32(Command(this.id, this.cmds['get_output_voltage']), callback);
    }

    isOutputEnabled(callback: (enabled: boolean) => void): void {
        this.client.readBool(Command(this.id, this.cmds['is_output_enabled']), callback);
    }
}

class VoltageControlApp {

    private outputToggle: HTMLInputElement;
    private voltageDisplay: HTMLSpanElement;
    private statusDisplay: HTMLSpanElement;

    constructor(document: Document, private voltageControl: VoltageControl) {
        this.outputToggle = <HTMLInputElement>document.getElementById('output-toggle');
        this.voltageDisplay = <HTMLSpanElement>document.getElementById('voltage-display');
        this.statusDisplay = <HTMLSpanElement>document.getElementById('status-display');

        this.setupEventListeners();
        this.updateStatus();
    }

    private setupEventListeners(): void {
        this.outputToggle.addEventListener('change', (event) => {
            const enabled = (<HTMLInputElement>event.target).checked;
            if (enabled) {
                this.voltageControl.setTestVoltage(); // Sets to 0.5V and enables
            } else {
                this.voltageControl.disableTestVoltage(); // Disables output
            }
            setTimeout(() => this.updateStatus(), 100); // Update status after short delay
        });
    }

    private updateStatus(): void {
        // Get both the output voltage and enabled status
        this.voltageControl.isOutputEnabled((enabled) => {
            this.outputToggle.checked = enabled;
            this.statusDisplay.textContent = enabled ? 'Enabled' : 'Disabled';
            this.statusDisplay.className = enabled ? 'label label-success' : 'label label-default';

            if (enabled) {
                // If enabled, show the actual voltage
                this.voltageControl.getOutputVoltage((voltage) => {
                    this.voltageDisplay.textContent = voltage.toFixed(3);
                });
            } else {
                // If disabled, show 0.000 (the actual DAC output)
                this.voltageDisplay.textContent = '0.000';
            }
        });

        // Update status every 500ms
        setTimeout(() => this.updateStatus(), 500);
    }
} 