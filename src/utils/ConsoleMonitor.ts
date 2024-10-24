type HistoryLog = {
    aparence: string,
    message: string,
    classes: string[]
}

export default class ConsoleMonitor{
    private lines:string;
    private history:HistoryLog[];

    constructor(){
        this.lines = '';
        this.history = [];
    }

    public asString(): string{
        return this.lines;
    }

    public getHistory(): HistoryLog[]{
        return this.history;
    }

    public log( message:string, aparence:'white', classes=[] ): void{
        console.log(message);

        this.lines = this.lines + message + '\n';

        this.history.push({
            aparence: aparence,
            message:  message,
            classes:  classes
        });
    }

    public reset(){
        this.lines = '';
        this.history = [];
    }
}