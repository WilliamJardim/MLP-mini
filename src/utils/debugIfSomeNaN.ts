
export default function debugIfSomeNaN( title:string, varToCheck:number[], callback?:(resultV:any)=>void ): any{
    let nanValues:number[] = [];
    let hasNaN:boolean = false;

    varToCheck.forEach( (val, valIndex)=>{
        if( isNaN(val) || !isFinite(val) ){
            nanValues.push( valIndex );
            console.warn( title, 'NaN', valIndex, 'please insert debugger' );
            hasNaN = true;
        }
    });

    let result:any = { hasNaN: hasNaN, values: nanValues };

    if(hasNaN && callback){
        callback(result);
    }

    return result;
}