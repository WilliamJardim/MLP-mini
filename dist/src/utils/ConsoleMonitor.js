export default class ConsoleMonitor {
    constructor(config) {
        this.lines = '';
        this.history = [];
        this.config = config;
        this.name = config.name;
    }
    getConsoleName() {
        return this.name;
    }
    asString() {
        return this.lines;
    }
    getHistory() {
        return this.history;
    }
    push(info) {
        this.getHistory().push(info);
    }
    integrate(from) {
        this.isIntegrator = true;
        //Para cada console vinculado
        for (let i = 0; i < from.length; i++) {
            //Extrai as informações e acrescenta elas na lista
            let currentLogs = from[i].getHistory();
            let consoleName = from[i].getConsoleName();
            //this.log(`CONSOLE: ${consoleName}`);
            currentLogs.forEach((info) => {
                this.push(Object.assign({}, info));
            });
        }
    }
    log(message, aparence = 'white', classes = []) {
        console.log(message);
        this.lines = this.lines + message + '\n';
        this.history.push({
            aparence: aparence,
            message: message,
            classes: classes
        });
    }
    reset() {
        this.lines = '';
        this.history = [];
    }
}
