class App {

    private imports: Imports;
    private voltageControl: VoltageControl;
    private voltageControlApp: VoltageControlApp;

    constructor(window: Window, document: Document, ip: string) {
        let sockpoolSize: number = 5;
        let client = new Client(ip, sockpoolSize);

        window.addEventListener('HTMLImportsLoaded', () => {
            client.init( () => {
                this.imports = new Imports(document);
                this.voltageControl = new VoltageControl(client);
                this.voltageControlApp = new VoltageControlApp(document, this.voltageControl);
            });
        }, false);

        window.onbeforeunload = () => { client.exit(); };
    }
}

let app = new App(window, document, location.hostname); 